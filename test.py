from collections import OrderedDict
import torch

# Ordered dictionary of worker weights
worker_weights = OrderedDict([
    ('worker_1', OrderedDict([
        ('conv1.weight', torch.tensor([[[[-0.0400, -0.1436,  0.0389, -0.0735, -0.1771],
                                        [-0.2107,  0.1206, -0.1566,  0.0306,  0.1039]]]])),
        ('conv1.bias', torch.tensor([0.1606,  0.0609,  0.0679, -0.1001, -0.0624,  0.0428,  0.0640, -0.1100,
                                     0.1774, -0.1197])),
        ('conv2.weight', torch.tensor([[[[-6.7766e-03,  7.7188e-03, -6.6041e-04,  4.3813e-02, -5.8251e-02],
                                         [-3.9509e-02, -4.6059e-02,  1.5709e-02,  3.3179e-02, -2.5063e-02],
                                         [-1.2974e-03, -4.2207e-02, -3.8424e-02,  1.0624e-02,  5.3539e-02]]]])),
        ('conv2.bias', torch.tensor([0.0570, -0.0075,  0.0585, -0.0526,  0.0455,  0.0006, -0.0074,  0.0479,
                                     0.0219,  0.0090,  0.0256,  0.0038, -0.0227, -0.0142, -0.0745, -0.0051,
                                     0.0017, -0.0121, -0.0284, -0.0642])),
        ('fc1.weight', torch.tensor([[-0.0056, -0.0584,  0.0261, -0.0230, -0.0526,  0.0234],
                                     [0.0501,  0.0183, -0.0487,  0.0160,  0.0013, -0.0263]])),
        ('fc1.bias', torch.tensor([-0.0007, -0.0057, -0.0525, -0.0326, -0.0539, -0.0352, -0.0107,  0.0376])),
        ('fc2.bias', torch.tensor([-0.0578,  0.0399,  0.0653,  0.0657, -0.0738, -0.1413,  0.0518, -0.1129,
                                    0.1070, -0.0945]))
    ])),
    ('worker_2', OrderedDict([
        ('conv1.weight', torch.tensor([[[[-0.0400, -0.1436,  0.0389, -0.0735, -0.1771],
                                        [-0.2107,  0.1206, -0.1566,  0.0306,  0.1039]]]])),
        ('conv1.bias', torch.tensor([0.1606,  0.0609,  0.0679, -0.1001, -0.0624,  0.0428,  0.0640, -0.1100,
                                     0.1774, -0.1197])),
        ('conv2.weight', torch.tensor([[[[-6.7766e-03,  7.7188e-03, -6.6041e-04,  4.3813e-02, -5.8251e-02],
                                         [-3.9509e-02, -4.6059e-02,  1.5709e-02,  3.3179e-02, -2.5063e-02],
                                         [-1.2974e-03, -4.2207e-02, -3.8424e-02,  1.0624e-02,  5.3539e-02]]]])),
        ('conv2.bias', torch.tensor([0.0570, -0.0075,  0.0585, -0.0526,  0.0455,  0.0006, -0.0074,  0.0479,
                                     0.0219,  0.0090,  0.0256,  0.0038, -0.0227, -0.0142, -0.0745, -0.0051,
                                     0.0017, -0.0121, -0.0284, -0.0642])),
        ('fc1.weight', torch.tensor([[-0.0056, -0.0584,  0.0261, -0.0230, -0.0526,  0.0234],
                                     [0.0501,  0.0183, -0.0487,  0.0160,  0.0013, -0.0263]])),
        ('fc1.bias', torch.tensor([-0.0007, -0.0057, -0.0525, -0.0326, -0.0539, -0.0352, -0.0107,  0.0376])),
        ('fc2.bias', torch.tensor([-0.0578,  0.0399,  0.0653,  0.0657, -0.0738, -0.1413,  0.0518, -0.1129,
                                    0.1070, -0.0945]))
    ]))
])

# Average the weights
averaged_weights = OrderedDict()
for layer_key in worker_weights['worker_1']:
    layer_weights = [worker_weights[worker][layer_key] for worker in worker_weights]
    averaged_weights[layer_key] = torch.stack(layer_weights).mean(dim=0)

print("Average weights",averaged_weights)
# # Create a new model with the averaged weights
# updated_model = YourModelClass()  # Replace YourModelClass with your actual model class
# updated_model.load_state_dict(averaged_weights)
