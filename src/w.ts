/// <reference path="TensorOperationsList.ts" />

const [t1, t2,t3] = [new Tensor([[1, 2, 3], [1, 0, 0], [1, 0, 0]]), new Tensor([[1, 2, 3], [1, 7, 0], [1, 9, 0]]), new Tensor([[1, 2, 3], [1, 7, 0], [1, 9, 0]])];

let j = new Multiply()([t1, t2,t3])
// j = new Max()([j,new Tensor([0])])
let mean = new Mean()([j])
mean.backward()
console.log(t1.gradientHandler.gradient);