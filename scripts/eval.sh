export OPENAI_API_KEY=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
num_frames=16
test_ratio=1

model_dir=MODELS/pllava-7b
weight_dir=MODELS/pllava-7b

lora_alpha=14
selected_layers=(10)
alphas=(0.4)
taus=(0.8)
temporal_segment_ratios=(0.25)
cluster_ratios=(0.5)

for alpha in "${alphas[@]}"; do
  for selected_layer in "${selected_layers[@]}"; do
    for tau in "${taus[@]}"; do
      for temporal_segment_ratio in "${temporal_segment_ratios[@]}"; do
        for cluster_ratio in "${cluster_ratios[@]}"; do
          # 执行命令
          SAVE_DIR=test_results/pllava-7b-lora${lora_alpha}-threshold${tau}-layer${selected_layer}-alpha${alpha}-temporal-segment-ratio-${temporal_segment_ratio}-cluster-ratio-${cluster_ratio}
          mkdir -p "${SAVE_DIR}"
          conv_mode=eval_mvbench
          python -m tasks.eval.mvbench.pllava_eval_mvbench \
              --pretrained_model_name_or_path ${model_dir} \
              --save_path ${SAVE_DIR}/mvbench \
              --num_frames ${num_frames} \
              --use_lora \
              --lora_alpha ${lora_alpha} \
              --top_p 1.0 \
              --temperature 1.0 \
              --weight_dir ${weight_dir} \
              --pooling_shape 16-12-12 \
              --conv_mode ${conv_mode} \
              --selected_layer ${selected_layer} \
              --alpha ${alpha} \
              --tau ${tau} \
              --temporal_segment_ratio ${temporal_segment_ratio} \
              --cluster_ratio ${cluster_ratio}
        done
      done
    done
  done
done

lora_alpha=14
selected_layers=(10)
alphas=(0.4)
taus=(0.8)
temporal_segment_ratios=(0.25)
cluster_ratios=(0.5)

for alpha in "${alphas[@]}"; do
  for selected_layer in "${selected_layers[@]}"; do
    for tau in "${taus[@]}"; do
      for temporal_segment_ratio in "${temporal_segment_ratios[@]}"; do
        for cluster_ratio in "${cluster_ratios[@]}"; do
          # 执行命令
          SAVE_DIR=test_results/pllava-7b-lora${lora_alpha}-threshold${tau}-layer${selected_layer}-alpha${alpha}-temporal-segment-ratio-${temporal_segment_ratio}-cluster-ratio-${cluster_ratio}
          mkdir -p "${SAVE_DIR}"
          conv_mode=eval_videomme
          python -m tasks.eval.videomme.pllava_eval_videomme \
              --pretrained_model_name_or_path ${model_dir} \
              --save_path ${SAVE_DIR}/videomme \
              --num_frames ${num_frames} \
              --use_lora \
              --lora_alpha ${lora_alpha} \
              --top_p 1.0 \
              --temperature 1.0 \
              --weight_dir ${weight_dir} \
              --pooling_shape 16-12-12 \
              --conv_mode ${conv_mode} \
              --selected_layer ${selected_layer} \
              --alpha ${alpha} \
              --tau ${tau} \
              --temporal_segment_ratio ${temporal_segment_ratio} \
              --cluster_ratio ${cluster_ratio}
        done
      done
    done
  done
done

lora_alpha=14
selected_layers=(10)
alphas=(0.4)
taus=(0.8)
temporal_segment_ratios=(0.25)
cluster_ratios=(0.5)

for alpha in "${alphas[@]}"; do
  for selected_layer in "${selected_layers[@]}"; do
    for tau in "${taus[@]}"; do
      for temporal_segment_ratio in "${temporal_segment_ratios[@]}"; do
        for cluster_ratio in "${cluster_ratios[@]}"; do
            # 执行命令
            SAVE_DIR=test_results/pllava-7b-lora${lora_alpha}-threshold${tau}-layer${selected_layer}-alpha${alpha}-temporal-segment-ratio-${temporal_segment_ratio}-cluster-ratio-${cluster_ratio}
            mkdir -p "${SAVE_DIR}"
            conv_mode=eval_mvbench
            python -m tasks.eval.egoshcema.pllava_eval_egoschema \
                --pretrained_model_name_or_path ${model_dir} \
                --save_path ${SAVE_DIR}/egoschema \
                --num_frames ${num_frames} \
                --use_lora \
                --lora_alpha ${lora_alpha} \
                --top_p 1.0 \
                --temperature 1.0 \
                --weight_dir ${weight_dir} \
                --pooling_shape 16-12-12 \
                --conv_mode ${conv_mode} \
                --selected_layer ${selected_layer} \
                --alpha ${alpha} \
                --tau ${tau} \
                --temporal_segment_ratio ${temporal_segment_ratio} \
                --cluster_ratio ${cluster_ratio}
        done
      done
    done
  done
done


lora_alpha=4
selected_layers=(5)
alphas=(0.4)
taus=(0.8)
temporal_segment_ratios=(0.25)
cluster_ratios=(0.5)

for alpha in "${alphas[@]}"; do
  for selected_layer in "${selected_layers[@]}"; do
    for tau in "${taus[@]}"; do
      for temporal_segment_ratio in "${temporal_segment_ratios[@]}"; do
        for cluster_ratio in "${cluster_ratios[@]}"; do
          # 执行命令
          SAVE_DIR=test_results/pllava-7b-lora${lora_alpha}-threshold${tau}-layer${selected_layer}-alpha${alpha}-temporal-segment-ratio-${temporal_segment_ratio}-cluster-ratio-${cluster_ratio}
          mkdir -p "${SAVE_DIR}"
          conv_mode=eval_vcgbench
          python -m tasks.eval.vcgbench.pllava_eval_vcgbench \
              --pretrained_model_name_or_path ${model_dir} \
              --save_path ${SAVE_DIR}/vcgbench \
              --num_frames ${num_frames} \
              --weight_dir ${weight_dir} \
              --pooling_shape 16-12-12 \
              --test_ratio ${test_ratio} \
              --use_lora \
              --lora_alpha ${lora_alpha} \
              --selected_layer ${selected_layer} \
              --alpha ${alpha} \
              --tau ${tau} \
              --temporal_segment_ratio ${temporal_segment_ratio} \
              --cluster_ratio ${cluster_ratio}
        done
      done
    done
  done
done