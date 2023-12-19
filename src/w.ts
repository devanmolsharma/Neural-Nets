/// <reference path="TensorOperationsList.ts" />
/// <reference path="Linear.ts" />
/// <reference path="Sequential.ts" />
/// <reference path="SGD.ts" />

let s = new Sequential();

s.add(new Linear(1, 4,'relu'))
s.add(new Linear(4, 1,'relu'))
s.add(new Linear(1, 1,'relu'))
const optimiser = new SGD(s.getParameters(), {lr:1e-4});

for (let i = 0; i < 1000; i++) {
    const out = s.forward([new Tensor([[2]])])
    const expected = new Tensor([[4]]);
    let loss = new Subtract()([out, expected]);
    loss = new Mean()([loss])
    loss.backward();
    optimiser.step();
    console.log(loss.value);

}

console.log(s.toJson());