/// <reference path="TensorOperationsList.ts" />
/// <reference path="Linear.ts" />

let l = new Linear(3, 10);
let mean  = new Mean()([l.forward(new Tensor([[1,2,3]]))]);
let loss = new Subtract()([mean, new Tensor([2])])
loss.backward();
console.log(l.parameters.get('weights').gradientHandler.gradient);