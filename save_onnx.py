import torch
import torch.onnx
import model.model_builder as model_builder
import dataset.dataset_builder as dataset_builder

device = 'cuda'

model_path = 'output/base/amdm_motionvivid_004'
model_config_file = 'output/base/amdm_motionvivid_004/config.yaml'

dataset = dataset_builder.build_dataset(model_config_file, load_full_dataset=True)
#model = torch.load('{}/model.pt'.format(model_path))
#torch.save(model.state_dict(), '{}/model_weights_new_test.pth'.format(model_path))

amdm = model_builder.build_model(model_config_file, dataset, device)
amdm.load_state_dict(torch.load('{}/model_param.pth'.format(model_path)))
use_ema = amdm.use_ema
sa_model = amdm.ema_diffusion if use_ema else amdm.diffusion
print(sa_model.betas)
print(sa_model.alphas)
print(sa_model.reciprocal_sqrt_alphas)
noise_model = sa_model.model
time_emb = sa_model.time_mlp
style_emb = sa_model.style_emb

#model.eval()

last_frame = torch.randn(1, sa_model.frame_dim).to(device)
print(last_frame.shape)
noise = torch.randn(1, sa_model.frame_dim).to(device)
time_step = torch.randint(0, sa_model.T, (1,)).to(device)
style = torch.zeros(1, sa_model.num_styles).to(device)

input_tuple = (last_frame, noise, time_step, style)

temb = time_emb(time_step)
semb = style_emb(style)
print(temb.shape, semb.shape)

latent = torch.cat((temb, semb), dim=-1)
result = noise_model(last_frame, noise, latent)

input_names = ['last_frame', 'noise', 'latent']
output_names = ['output']

# torch.onnx.export(sa_model, *dummy_x, dynamo)
torch.onnx.export(time_emb, (time_step,), './onnx/time_emb.onnx', verbose=True, input_names=['ts'], 
                  output_names=['time_emb'], opset_version=15, export_params=True)
torch.onnx.export(style_emb, (style,), './onnx/style_emb.onnx', verbose=True, input_names=['style'], 
                  output_names=['style_emb'], opset_version=15, export_params=True)
torch.onnx.export(noise_model, (last_frame, noise, latent), './onnx/denoise.onnx', verbose=True, input_names=input_names,
                  output_names=output_names, opset_version=15, export_params=True)
# torch.onnx.dynamo_export(sa_model,*dummy_x).save("amdm.onnx")                       
                  