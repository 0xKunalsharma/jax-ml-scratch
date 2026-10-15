import jax as jp
import jax.numpy as jnp
from jax import random
import numpy as np
from torchvision import datasets, transforms

input_dim  = 28 * 28   # 784
hidden_dim = 128
output_dim = 10
BATCH      = 64
EPOCHS     = 3
LR         = 0.1

def mlp_init(rng):
    k1, k2 = random.split(rng, 2)
    scale1 = (1.0 / input_dim)  ** 0.5   # ≈ 0.036
    scale2 = (1.0 / hidden_dim) ** 0.5   # ≈ 0.088
    mlp_1 = random.uniform(k1, (input_dim,  hidden_dim),
                           minval=-scale1, maxval=scale1, dtype=jnp.float32)
    mlp_2 = random.uniform(k2, (hidden_dim, output_dim),
                           minval=-scale2, maxval=scale2, dtype=jnp.float32)
    return mlp_1, mlp_2

def bias_init():
    return jnp.zeros((hidden_dim,)), jnp.zeros((output_dim,))

print("Loading MNIST...")
transform = transforms.Compose([
    transforms.ToTensor(),                       # 0..255 → 0..1
    transforms.Normalize((0.1307,), (0.3081,)), # ARENA's (x - 0.1307) / 0.3081
])
train_ds = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

def to_numpy(ds):
    n = len(ds)
    x = np.empty((n, 1, 28, 28), dtype=np.float32)
    y = np.empty((n,),            dtype=np.int32)
    for i, (img, label) in enumerate(ds):
        x[i] = img.numpy(); y[i] = label
    return x, y

train_x, train_y = to_numpy(train_ds)
test_x,  test_y  = to_numpy(test_ds)
print(f"  train: {train_x.shape}  test: {test_x.shape}")

rng = random.key(0)
rng, rng_init = random.split(rng, 2)
mlp_1, mlp_2 = mlp_init(rng_init)
bias_1, bias_2 = bias_init()

params = {"mlp_1": mlp_1, "mlp_2": mlp_2,
          "bias_1": bias_1, "bias_2": bias_2}

@jp.jit
def relu(x):
    return jnp.maximum(0, x)

@jp.jit
def softmax(x):
    exp_x = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)

@jp.jit
def forwardpass(params, x):
    z_1 = x @ params["mlp_1"] + params["bias_1"]    
    A_1 = relu(z_1)
    z_2 = A_1 @ params["mlp_2"] + params["bias_2"]   
    return z_2                                        

def loss_fn(params, x, y):
    logits    = forwardpass(params, x)
    log_probs = jp.nn.log_softmax(logits)          
    one_hot   = jp.nn.one_hot(y, num_classes=10)
    return -jnp.sum(one_hot * log_probs) / x.shape[0]

@jp.jit
def train_step(params, x, y, lr):
    grads = jp.grad(loss_fn)(params, x, y)           
    return jp.tree_util.tree_map(
        lambda w, g: w - lr * g, params, grads
    )

@jp.jit
def batch_accuracy(params, x, y):
    logits = forwardpass(params, x)
    preds  = jnp.argmax(logits, axis=1)
    return jnp.mean(preds == y)

def evaluate(params, x, y, batch_size=1000):
    accs = []
    for i in range(0, x.shape[0], batch_size):
        xb = jnp.asarray(x[i:i+batch_size])
        yb = jnp.asarray(y[i:i+batch_size])
        accs.append(batch_accuracy(params, xb, yb))
    return float(jnp.mean(jnp.stack(accs)))

print("\nBefore training — one MNIST image:")
sample_x = jnp.asarray(train_x[0:1].reshape(1, -1)) 
sample_y = int(train_y[0])
logits_before = forwardpass(params, sample_x)
probs_before  = softmax(logits_before)
print(f"  true label: {sample_y}")
print(f"  logits std: {float(jnp.std(logits_before)):.4f}     (was 122 with your old init)")
print(f"  probs:      {np.array2string(np.asarray(probs_before[0]), precision=3)}")
print(f"  predicted:  {int(jnp.argmax(logits_before, axis=1)[0])}")

print(f"\nTraining {EPOCHS} epochs, batch={BATCH}, lr={LR}...\n")
for epoch in range(1, EPOCHS + 1):
    perm = np.random.permutation(len(train_x))
    running_loss, n_batches = 0.0, len(train_x) // BATCH
    for b in range(n_batches):
        idx = perm[b*BATCH:(b+1)*BATCH]
        xb = jnp.asarray(train_x[idx].reshape(BATCH, -1))   # (64, 784)
        yb = jnp.asarray(train_y[idx])
        params = train_step(params, xb, yb, LR)
        running_loss += float(loss_fn(params, xb, yb))
        if b % 300 == 0:
            print(f"    epoch {epoch} batch {b:>4}  loss={float(loss_fn(params, xb, yb)):.4f}")
    avg_loss = running_loss / n_batches
    acc = evaluate(params, test_x.reshape(len(test_x), -1), test_y)
    print(f"  Epoch {epoch}/{EPOCHS}  train_loss={avg_loss:.4f}  test_accuracy={acc*100:.2f}%\n")

print("After training — same MNIST image:")
logits_after = forwardpass(params, sample_x)
probs_after  = softmax(logits_after)
print(f"  true label: {sample_y}")
print(f"  logits std: {float(jnp.std(logits_after)):.4f}")
print(f"  probs:      {np.array2string(np.asarray(probs_after[0]), precision=3)}")
print(f"  predicted:  {int(jnp.argmax(logits_after, axis=1)[0])}")

print("\nDone. Same code you wrote — just with the bugs pulled out.")
