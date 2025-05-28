from hqq.core.quantize import BaseQuantizeConfig

def get_quant_config(model):
    quant_config = {}
    
    config_4bits = BaseQuantizeConfig(nbits=4, group_size=64)
    config_8bits = BaseQuantizeConfig(nbits=8, group_size=64)
    
    for i in range(model.config.num_hidden_layers):
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = config_8bits
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = config_8bits
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = config_8bits
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = config_8bits
        
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = config_4bits if i > 3 else config_8bits
        quant_config[f'model.layers.{i}.mlp.up_proj'] = config_4bits if i > 3 else config_8bits
        quant_config[f'model.layers.{i}.mlp.down_proj'] = config_4bits if i > 3 else config_8bits
        
    return quant_config