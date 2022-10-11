import torch
import schnetpack as spk
import os
from schnetpack.datasets import MD17
from torch.optim import Adam
import schnetpack.train as trn
import numpy as np
import matplotlib.pyplot as plt
from schnetpack.new.EnbeddingNet.encoder import Encoder

forcetut = './ethanol_newEmbedding'
ethanol = '../ethanol'
new_embedding = "Triplet.pth"
batch = 100
train_sample = 10000
val_sample = 2000
if not os.path.exists(forcetut):
    os.makedirs(forcetut)

ethanol_data = MD17(os.path.join(ethanol, 'ethanol.db'), molecule='ethanol')

atoms, properties = ethanol_data.get_properties(0)

print('Loaded properties:\n', *['{:s}\n'.format(i) for i in properties.keys()])

print('Forces:\n', properties[MD17.forces])
print('Shape:\n', properties[MD17.forces].shape)

train, val, test = spk.train_test_split(
        data=ethanol_data,
        num_train=train_sample,
        num_val=val_sample,
        split_file=os.path.join(forcetut, "split.npz"),
    )

train_loader = spk.AtomsLoader(train, batch_size=batch, shuffle=True, num_workers=6, pin_memory=True)
val_loader = spk.AtomsLoader(val, batch_size=batch, num_workers=6, pin_memory=True)

means, stddevs = train_loader.get_statistics(
    spk.datasets.MD17.energy, divide_by_atoms=True
)

print('Mean atomization energy / atom:      {:12.4f} [kcal/mol]'.format(means[MD17.energy][0]))
print('Std. dev. atomization energy / atom: {:12.4f} [kcal/mol]'.format(stddevs[MD17.energy][0]))

n_features = 64

schnet = spk.representation.SchNet(
    n_atom_basis=n_features,
    n_filters=n_features,
    n_gaussians=25,
    n_interactions=6,
    cutoff=5.,
    cutoff_network=spk.nn.cutoff.CosineCutoff,
    Gass_true_atten_flase=False
)

energy_model = spk.atomistic.Atomwise(
    n_in=n_features,
    property=MD17.energy,
    mean=means[MD17.energy],
    stddev=stddevs[MD17.energy],
    derivative=MD17.forces,
    negative_dr=True,
)

model = spk.AtomisticModel(representation=schnet, output_modules=energy_model)

# tradeoff
rho_tradeoff = 0.01

# loss function
def loss(batch, result):
    # compute the mean squared error on the energies
    diff_energy = batch[MD17.energy]-result[MD17.energy]
    err_sq_energy = torch.mean(diff_energy ** 2)

    # compute the mean squared error on the forces
    diff_forces = batch[MD17.forces]-result[MD17.forces]
    err_sq_forces = torch.mean(diff_forces ** 2)

    # build the combined loss function
    err_sq = rho_tradeoff*err_sq_energy + (1-rho_tradeoff)*err_sq_forces

    return err_sq

optimizer = Adam(model.parameters(), lr=1e-3)

# set up metrics
metrics = [
    spk.metrics.MeanAbsoluteError(MD17.energy),
    spk.metrics.MeanAbsoluteError(MD17.forces)
]

# construct hooks
hooks = [
    trn.CSVHook(log_path=forcetut, metrics=metrics),
    trn.ReduceLROnPlateauHook(
        optimizer,
        patience=150, factor=0.8, min_lr=1e-6,
        stop_after_min=True
    )
]

trainer = trn.Trainer(
    model_path=forcetut,
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
)

# check if a GPU is available and use a CPU otherwise
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# determine number of epochs and train
n_epochs = 500
# load EmbeddingNew
em = torch.load(new_embedding)
print(em)
# start train
trainer.train(device=device, n_epochs=n_epochs, em_net=em)

# Load logged results
results = np.loadtxt(os.path.join(forcetut, 'log.csv'), skiprows=1, delimiter=',')

# Determine time axis
time = results[:, 0]-results[0, 0]

# Load the validation MAEs
energy_mae = results[:, 4]
forces_mae = results[:, 5]

# Get final validation errors
print('Validation MAE:')
print('    energy: {:10.3f} kcal/mol'.format(energy_mae[-1]))
print('    forces: {:10.3f} kcal/mol/\u212B'.format(forces_mae[-1]))

# Construct figure
plt.figure(figsize=(14, 5))

# Plot energies
plt.subplot(1, 2, 1)
plt.plot(time, energy_mae)
plt.title('Energy')
plt.ylabel('MAE [kcal/mol]')
plt.xlabel('Time [s]')

# Plot forces
plt.subplot(1, 2, 2)
plt.plot(time, forces_mae)
plt.title('Forces')
plt.ylabel('MAE [kcal/mol/\u212B]')
plt.xlabel('Time [s]')

plt.savefig('new_train.png')
