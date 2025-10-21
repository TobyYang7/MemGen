import os
from torch.utils.data import DataLoader
from datasets import Dataset
from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase
from trl import SFTTrainer, SFTConfig, GRPOConfig
from trl.models import unwrap_model_for_generation

from typing import Tuple, Dict, List
from tqdm import tqdm

from larm.task.base_runner import BaseRunner
from larm.common.registry import registry
from larm.common.config import Config
from larm.data.interactions.base_interaction import (
    InteractionConfig,   
    InteractionManager, 
    InteractionDataProto
)

from .memgen_model import LatentMemoryModel
from .trainer.weaver_grpo_trainer import WeaverGRPOTrainer
from .trainer.trigger_grpo_trainer import TriggerGRPOTrainer
from .utils import (
    fix_model_parameters, 
    open_model_parameters, 
    EvalConfig, 
    StaticEvalRecorder,
    DynamicEvalRecorder
)
from PIL import Image
import torch

@registry.register_runner("latmem")
class LatentMemoryRunner(BaseRunner):

    def __init__(
        self,
        model: LatentMemoryModel,
        processing_class: PreTrainedTokenizerBase,
        datasets_dict: Dict,  
        configs: Config,
        env_and_gens_dict: Dict 
    ):  
        super().__init__(
            model,
            processing_class,
            datasets_dict,  
            configs,
            env_and_gens_dict
        )
        # parse configs
        self._parse_configs(configs.run_cfg)

        # initialize envs and generation managers
        dataset_config = configs.datasets_cfg[self.dataset_name]
        self.env = self.env_cls(dataset_config)

        # partition datasets
        self.weaver_train_dataset, self.trigger_train_dataset = self._parse_train_dataset(self.dataset_dict["train"])
        self.valid_dataset = self.dataset_dict["valid"]
        self.test_dataset = self.dataset_dict["test"]
        
        self.weaver_train_dataset = self._filter_dataset(self.weaver_train_dataset)
        self.trigger_train_dataset = self._filter_dataset(self.trigger_train_dataset)
        self.valid_dataset = self._filter_dataset(self.valid_dataset)

        # initialize generation manager
        self.generation_manager: InteractionManager = self.gen_cls(
            self.processing_class, self.model, self.interaction_config
        )
        
        # VIS: build multimodal features if present in dataset
        if any(k in self.weaver_train_dataset.column_names for k in ("image_path",)):
            self.weaver_train_dataset = self._prepare_mm_features(self.weaver_train_dataset)
        if any(k in self.valid_dataset.column_names for k in ("image_path",)):
            self.valid_dataset = self._prepare_mm_features(self.valid_dataset)
        if any(k in self.test_dataset.column_names for k in ("image_path",)):
            self.test_dataset = self._prepare_mm_features(self.test_dataset)
    
    def _parse_train_dataset(self, train_dataset: Dataset) -> Tuple[Dataset, Dataset]:
        trigger_trainset_size = min(500, len(train_dataset))
        return train_dataset, train_dataset.select(range(trigger_trainset_size))

    def _filter_dataset(self, dataset: Dataset) -> Dataset:
        """Filter the dataset based on maximum sequence length.

        The maximum length depends on the training mode and method:
        - For Weaver SFT training: use `weaver_training_args.max_length`.
        - For Weaver GRPO training: use `weaver_training_args.max_prompt_length`.
        - For Trigger GRPO training: use `trigger_training_args.max_prompt_length`.

        Any sample exceeding the maximum length is filtered out.

        Args:
            dataset (Dataset): The input dataset to be filtered.

        Returns:
            Dataset: A filtered dataset containing only samples within the max length.
        """
        tokenizer = self.processing_class

        # Determine max length based on training mode
        max_len = 1024
        if self.train_weaver and self.train_weaver_method == "sft":
            max_len = self.weaver_training_args.max_length
        elif self.train_weaver and self.train_weaver_method == "grpo":
            max_len = self.weaver_training_args.max_prompt_length
        elif self.train_trigger and self.train_trigger_method == "grpo":
            max_len = self.trigger_training_args.max_prompt_length
        else:
            raise ValueError("Wrong training mode.")

        # Function to filter out samples exceeding max length
        def filter_func(sample):
            if "prompt" in sample and sample["prompt"] is not None:
                encoded = tokenizer(sample["prompt"], add_special_tokens=True)
                return len(encoded["input_ids"]) < max_len
            elif "messages" in sample and sample["messages"] is not None:
                conversation = tokenizer.apply_chat_template(sample["messages"][:2], tokenize=True)
                return len(conversation) < max_len
            return True 

        # Apply filtering
        dataset = dataset.filter(filter_func)

        return dataset
    
    def _prepare_mm_features(self, dataset: Dataset) -> Dataset:
        processor = self.processing_class  # AutoProcessor for VL models

        def _encode(example: Dict) -> Dict:
            prompt = example.get("prompt")
            completion = example.get("completion")
            image_path = example.get("image_path")

            image = None
            if image_path is not None and os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception:
                    image = None

            if image is not None:
                enc_prompt = processor(text=[prompt], images=[image], return_tensors="pt", padding=False)
                prompt_ids = enc_prompt["input_ids"][0]
                prompt_mask = enc_prompt["attention_mask"][0]
                pixel_values = enc_prompt.get("pixel_values")
                if pixel_values is not None:
                    pixel_values = pixel_values[0]
            else:
                enc_prompt = processor(text=[prompt], return_tensors="pt", padding=False)
                prompt_ids = enc_prompt["input_ids"][0]
                prompt_mask = enc_prompt["attention_mask"][0]
                pixel_values = None

            tokenizer = getattr(processor, "tokenizer", processor)
            enc_comp = tokenizer(text=[completion], add_special_tokens=False, return_tensors="pt")
            comp_ids = enc_comp["input_ids"][0]

            input_ids = torch.cat([prompt_ids, comp_ids], dim=0)
            attention_mask = torch.cat([prompt_mask, torch.ones_like(comp_ids)], dim=0)
            labels = torch.cat([torch.full_like(prompt_ids, -100), comp_ids.clone()], dim=0)

            image_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            if image is not None:
                vision_start_ids = tokenizer.encode("<|vision_start|>", add_special_tokens=False)
                vision_end_ids = tokenizer.encode("<|vision_end|>", add_special_tokens=False)
                ids_list = prompt_ids.tolist()

                def find_subseq(seq, sub):
                    n, m = len(seq), len(sub)
                    for i in range(0, n - m + 1):
                        if seq[i:i+m] == sub:
                            return i
                    return -1

                s_idx = find_subseq(ids_list, vision_start_ids) if len(vision_start_ids) > 0 else -1
                e_idx = find_subseq(ids_list, vision_end_ids) if len(vision_end_ids) > 0 else -1
                if s_idx != -1 and e_idx != -1 and e_idx >= s_idx:
                    image_token_mask[s_idx:e_idx+len(vision_end_ids)] = True

            out = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "image_token_mask": image_token_mask,
            }
            if pixel_values is not None:
                out["pixel_values"] = pixel_values
            return out

        dataset = dataset.map(_encode, remove_columns=[c for c in dataset.column_names if c not in ("prompt", "completion", "solution", "image_path")])
        return dataset
    
    # ===== train weaver =====
    def _create_weaver_trainer(self):

        # SFT Trainer
        if self.train_weaver_method == "sft":
            weaver_trainer = SFTTrainer(
                model=self.model,
                args=self.weaver_training_args,
                train_dataset=self.weaver_train_dataset,
                eval_dataset=self.valid_dataset,
                processing_class=self.processing_class,
            )
        
        # GRPO Trainer
        elif self.train_weaver_method == 'grpo':
            
            reward_funcs = []     # get reward funcs for GRPO Trainer
            for reward_name in self.weaver_reward_names:
                reward_funcs.append(self.env_cls.get_reward_func(reward_name))
            
            weaver_trainer = WeaverGRPOTrainer(
                model=self.model,
                reward_funcs=reward_funcs,
                args=self.weaver_training_args,
                train_dataset=self.weaver_train_dataset,
                eval_dataset=self.valid_dataset,
                processing_class=self.processing_class,
                env_class=self.env_cls,
                env_main_config=self.configs.datasets_cfg[self.dataset_name],
                generation_manager=self.generation_manager
            )
        else:
            raise ValueError("Unsupported weaver training method.")

        return weaver_trainer

    def _train_weaver(self):

        # fix trigger parameters
        fix_model_parameters(self.model.trigger)
        
        # train weaver
        weaver_trainer = self._create_weaver_trainer()
        weaver_trainer.train()
        weaver_trainer.save_model()   # save the best model
        
        # remove checkpoints and save weaver
        output_dir = weaver_trainer.args.output_dir
        self._remove_trainer_ckpts(output_dir)
        
        # open trigger parameters
        open_model_parameters(self.model.trigger)
    
    
    # ===== train trigger =====
    def _create_trigger_trainer(self):
        
        # get reward funcs for RL Trainer
        reward_funcs = [] 
        for reward_name in self.trigger_reward_names:
            reward_funcs.append(self.env_cls.get_reward_func(reward_name))
        
        # build trainer
        trigger_trainer = TriggerGRPOTrainer(
            model=self.model, 
            processing_class=self.processing_class, 
            train_dataset=self.trigger_train_dataset, 
            eval_dataset=self.valid_dataset, 
            reward_funcs=reward_funcs,
            args=self.trigger_training_args
        )

        return trigger_trainer
    
    def _train_trigger(self):

        # fix weaver parameters
        fix_model_parameters(self.model.weaver)

        # train trigger
        trigger_trainer = self._create_trigger_trainer()
        trigger_trainer.train()
        trigger_trainer.save_model()     # save the best model

        # remove checkpoints and save weaver
        output_dir = trigger_trainer.args.output_dir
        self._remove_trainer_ckpts(output_dir)
        
        # open trigger parameters
        open_model_parameters(self.model.weaver)
    
    # ===== train weaver/trigger =====
    def train(self):
        # train weaver
        if self.train_weaver:
            self._train_weaver()
        
        # train trigger
        if self.train_trigger:
            self._train_trigger()
    
    # ===== evaluate =====
    def _static_evaluate(self):
        
        accelerator = Accelerator()
        writer = self._create_tensorboard(mode="evaluate")
        
        batch_size = self.eval_config.batch_size
        output_dir = self.eval_config.output_dir
        generation_config = self.eval_config.generation_config
        generation_config.eos_token_id = self.processing_class.eos_token_id

        # prepare dataset and dataloader
        test_dataloader = accelerator.prepare(DataLoader(
            dataset=self.test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=lambda batch: batch  # use the identity function
        ))

        # prepare model
        model_wrapped = accelerator.prepare_model(model=self.model, evaluation_mode=True)
        model_wrapped.eval()
        
        # construct eval recorder
        test_funcs = [self.env_cls.get_reward_func("accuracy")]
        save_file = os.path.join(output_dir, "answer.json")
        recorder = StaticEvalRecorder(compute_metrics=test_funcs, writer=writer, log_file=save_file)
        
        # batch generation
        for test_batch in tqdm(test_dataloader):
            with unwrap_model_for_generation(
                model_wrapped, accelerator
            ) as unwrapped_model:
                # construct InteractionDataProto object
                prompts = [x["prompt"] for x in test_batch]
                prompt_inputs = self.processing_class(
                    text=prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=True
                )
                prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
                gen_batch = InteractionDataProto()
                gen_batch.batch["input_ids"] = prompt_ids.to(accelerator.device)
                gen_batch.batch["attention_mask"] = prompt_mask.to(accelerator.device)
                gen_batch.no_tensor_batch["initial_prompts"] = [x["prompt"] for x in test_batch]

                # generation manager
                self.generation_manager.actor_rollout_wg = unwrapped_model
                gen_output = self.generation_manager.run_agent_loop(gen_batch)
            
                # postprocess: 由 generation manager 保证 completion ids 的正确性
                completion_ids = gen_output.batch["responses"]
                completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

            # 转为 seperated examples
            recorder.record_batch(completions, test_batch)
        recorder.finalize()
        writer.close()


    def _dynamic_evaluate(self):
        
        def _set_batch_envs(batch: List) -> Tuple[List[str], List[str], List]:  # batch set envs
            system_prompts, init_user_prompts, envs = [], [], []
            for task_config in batch:
                env = self.env_cls(self.configs.datasets_cfg[self.dataset_name])
                system_prompt, init_user_prompt = env.set_env(task_config)

                system_prompts.append(system_prompt)
                init_user_prompts.append(init_user_prompt)
                envs.append(env)
            
            return system_prompts, init_user_prompts, envs
        
        def _build_data_proto(
            system_prompts: List[str], init_user_prompts: List[str], envs: List
        ) -> InteractionDataProto:
            messages = []
            for system_prmopt, init_user_prompt in zip(system_prompts, init_user_prompts):
                system_message = {"role": "system", "content": system_prmopt}
                user_message = {"role": "user", "content": init_user_prompt}
                init_messages = [system_message, user_message]
                messages.append(init_messages)

            data_proto = InteractionDataProto()
            data_proto.no_tensor_batch["init_prompts"] = messages
            data_proto.no_tensor_batch["envs"] = envs

            return data_proto
        
        # ===== body =====
        output_dir = self.eval_config.output_dir

        accelerator = Accelerator()
        writer = self._create_tensorboard(mode="evaluate") 
        save_file = os.path.join(output_dir, "conversations.txt")
        recorder = DynamicEvalRecorder(writer=writer, log_file=save_file)

        batch_size = self.eval_config.batch_size
        generation_config = self.eval_config.generation_config
        generation_config.eos_token_id = self.processing_class.eos_token_id
        
        # prepare dataset and dataloader
        test_dataloader = accelerator.prepare(DataLoader(
            dataset=self.test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=lambda batch: batch  # use the identity function
        ))

        # prepare model
        model_wrapped = accelerator.prepare_model(model=self.model, evaluation_mode=True)
        model_wrapped.eval()
        
        # batch generate
        for step, test_batch in tqdm(enumerate(test_dataloader)):
            with unwrap_model_for_generation(
                model_wrapped, accelerator
            ) as unwrapped_model:
                system_prompts, init_user_prompts, envs = _set_batch_envs(test_batch) 
                input_data_proto = _build_data_proto(system_prompts, init_user_prompts, envs)
                
                self.generation_manager.actor_rollout_wg = unwrapped_model
                outputs: InteractionDataProto = self.generation_manager.run_agent_loop(input_data_proto)
                
                inter_histories = outputs.no_tensor_batch["inter_histories"]
                inter_context = self.processing_class.apply_chat_template(inter_histories, tokenize=False)

            # batch record
            rewards = []
            for env in input_data_proto.no_tensor_batch["envs"]:
                reward = env.feedback()
                rewards.append(reward)
            
            recorder.record_batch(inter_context, rewards)
        
        recorder.finalize()
        writer.close()

    
    # ===== runner configs =====
    def _parse_configs(self, configs):
        """build configs
        - weaver training config
        - trigger training config
        - interaction config
        - evaluatoin config
        """
        self.save_dir = configs.get("save_dir")
        use_tensorboard = configs.get("use_wandb")
        
        # weaver configs
        self.train_weaver = configs.get("train_weaver", True)
        if self.train_weaver:
            self.train_weaver_method = configs.get("train_weaver_method", "sft")
            weaver_save_dir = os.path.join(self.save_dir, "weaver")
            weaver_config = configs.get("weaver", {})
            
            # train weaver with sft
            if self.train_weaver_method == "sft":
                sft_config = weaver_config.get("sft", {})
                weaver_args_dict = self._parse_common_training_args(sft_config, weaver_save_dir, use_tensorboard)
                self.weaver_training_args = SFTConfig(**weaver_args_dict)
            
            # train weaver with grpo
            elif self.train_weaver_method == "grpo":
                grpo_config = weaver_config.get("grpo", {})
                weaver_args_dict = self._parse_common_training_args(grpo_config, weaver_save_dir, use_tensorboard, is_grpo=True)
                self.weaver_reward_names = weaver_args_dict.pop("reward_names")
                
                self.weaver_training_args = GRPOConfig(**weaver_args_dict)

            else:
                raise ValueError("Unsupported weaver training mode")

        # Trigger configs
        self.train_trigger = configs.get("train_trigger", False)
        if self.train_trigger:
            self.train_trigger_method = configs.get("train_trigger_method", "grpo")
            trigger_save_dir = os.path.join(self.save_dir, "trigger")
            trigger_config = configs.get("trigger", {}) 

            if self.train_trigger_method == "grpo":
                grpo_config = trigger_config.get("grpo", {})
                trigger_args_dict = self._parse_common_training_args(grpo_config, trigger_save_dir, use_tensorboard, is_grpo=True)
                self.trigger_reward_names = trigger_args_dict.pop("reward_names")
                
                self.trigger_training_args = GRPOConfig(**trigger_args_dict)
            else:
                raise ValueError("Unsupported weaver training mode")

        # Interaction configs
        generation_configs = configs.get("generation", {})
        self.interaction_config = InteractionConfig(
            max_turns=generation_configs.get("max_turns", 30),
            max_start_length=generation_configs.get("max_start_length", 1024),
            max_prompt_length=generation_configs.get("max_prompt_length", 4096),
            max_response_length=generation_configs.get("max_response_length", 512),
            max_obs_length=generation_configs.get("max_obs_length", 512),
            do_sample=generation_configs.get("do_sample", False),
            temperature=generation_configs.get("temperature", 1.0)
        )
        
        # Evaluation configs
        eval_dir = os.path.join(self.save_dir, "evaluate")
        eval_batch_size = generation_configs.get("eval_batch_size", 32)
        self.eval_config = EvalConfig(
            output_dir=eval_dir, batch_size=eval_batch_size, generation_config=self.interaction_config
        )

        # Align GRPO generation configuration with the interaction configuration:
        # All generation-related settings are controlled by the interaction config.
        if (self.train_weaver and self.train_weaver_method == "grpo"):
            self.weaver_training_args.max_prompt_length = self.interaction_config.max_start_length
            self.weaver_training_args.max_completion_length = self.interaction_config.max_response_length
            self.weaver_training_args.temperature = self.interaction_config.temperature
        elif (self.train_trigger and self.train_trigger_method == "grpo"):
            self.trigger_training_args.max_prompt_length = self.interaction_config.max_start_length
            self.trigger_training_args.max_completion_length = self.interaction_config.max_response_length
            self.trigger_training_args.temperature = self.interaction_config.temperature
       
    def _parse_common_training_args(self, config_dict, output_dir, use_tensorboard, is_grpo=False):
        batch_size = config_dict.get("batch_size", 4)
        max_epochs = config_dict.get("max_epochs", 2)
        grad_accum_steps = config_dict.get("grad_accum_steps", 1)

        optim = config_dict.get("optim", "adamw_torch")
        lr = config_dict.get("lr", 1e-5)
        scheduler = config_dict.get("schedular", "cosine")
        warmup_ratio = config_dict.get("warmup_ratio", 0.1)

        logging_strategy = config_dict.get("logging_strategy", "steps")
        logging_steps = config_dict.get("logging_steps", 1) if logging_strategy == "steps" else None
        
        eval_strategy = config_dict.get("eval_strategy", "steps")
        eval_steps = config_dict.get("eval_steps", 200) if eval_strategy == "steps" else None
        
        save_strategy = config_dict.get("save_strategy", "steps")
        save_steps = config_dict.get("save_steps", 200) if save_strategy == "steps" else None

        # common args dict
        args_dict = {
            "output_dir": output_dir,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "num_train_epochs": max_epochs,
            "gradient_accumulation_steps": grad_accum_steps,
            "optim": optim,
            "learning_rate": lr,
            "lr_scheduler_type": scheduler,
            "warmup_ratio": warmup_ratio,
            "logging_strategy": logging_strategy,
            "logging_steps": logging_steps,
            "save_strategy": save_strategy,
            "save_steps": save_steps,
            "eval_strategy": eval_strategy,
            "eval_steps": eval_steps,
            "report_to": ["tensorboard"] if use_tensorboard else [],   # <-- set to tensorboard
            "remove_unused_columns": False,
            "load_best_model_at_end": True,
            "bf16": True,
        }
        
        # add grpo specific args
        if is_grpo:
            args_dict.update({
                "num_generations": config_dict.get("num_generations", 16),
                "num_iterations": config_dict.get("num_iterations", 1),
                "beta": config_dict.get("beta", 0.0),
                "loss_type": config_dict.get("loss_type", "grpo"),
                "max_prompt_length": config_dict.get("max_prompt_length", 1024),
                "max_completion_length": config_dict.get("max_completion_length", 512),
            })

            rewards = config_dict.get("reward_funcs", [])
            reward_weights = [r["weight"] for r in rewards]
            reward_names = [r["name"] for r in rewards]  
            
            args_dict.update({
                "reward_weights": reward_weights,
                "reward_names": reward_names
            })
        # add sft specific args
        else:
            args_dict.update({
                "max_length": config_dict.get("max_length", 1024),
                "assistant_only_loss": config_dict.get("assistant_only_loss", True)
            })
        
        return args_dict
    