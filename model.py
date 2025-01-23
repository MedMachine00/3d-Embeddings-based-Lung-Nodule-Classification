import torch
import torch.nn as nn
import resnet
import os 
def generate_model(opt):
    # Ensure model name matches expected value
    assert opt.model.lower() in ['resnet'], f"Unsupported model type: {opt.model}"

    # Validate model depth
    assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200], \
        f"Unsupported model depth: {opt.model_depth}"

    # Handle missing embedding_dim gracefully
    embedding_dim = getattr(opt, 'embedding_dim', 128)  # Default to 128

    # Map model depth to corresponding ResNet function
    model_mapping = {
        10: resnet.resnet10,
        18: resnet.resnet18,
        34: resnet.resnet34,
        50: resnet.resnet50,
        101: resnet.resnet101,
        152: resnet.resnet152,
        200: resnet.resnet200
    }

    # Instantiate the ResNet model
    model = model_mapping[opt.model_depth](
        sample_input_W=opt.input_W,
        sample_input_H=opt.input_H,
        sample_input_D=opt.input_D,
        shortcut_type=opt.resnet_shortcut,
        no_cuda=opt.no_cuda,
        embedding_dim=embedding_dim
    )

    # Set device and enable DataParallel if using multiple GPUs
    if not opt.no_cuda:
        if isinstance(opt.gpu_id, list) and len(opt.gpu_id) > 1:
            model = nn.DataParallel(model, device_ids=opt.gpu_id).cuda()
        else:
            if isinstance(opt.gpu_id, list) and len(opt.gpu_id) > 0:
                torch.cuda.set_device(opt.gpu_id[0])
            model = model.cuda()

    # Load pretrained weights
    if opt.phase != 'test' and opt.pretrain_path:
        print(f'Loading pretrained model from {opt.pretrain_path}')
        state_dict = torch.load(opt.pretrain_path, map_location='cpu')

        # Remove unnecessary keys like embedding_fc if not part of architecture
        state_dict = {k: v for k, v in state_dict.items() if 'embedding_fc' not in k}
        model.load_state_dict(state_dict, strict=False)

    # Return the model and its parameters
    return model, model.parameters()
