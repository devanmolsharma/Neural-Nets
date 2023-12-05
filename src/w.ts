/// <reference path="TensorOperationsList.ts" />

const [t1, t2] = [new Tensor([[1, 2, 3],[1, 0, 0],[1, 0, 0]]), new Tensor([[1, 2], [3, 4], [3, 4]])];

let j = new Matmul()([t1, t2])
let mean = new Mean()([j])
mean.backward()
console.log(t1.gradientHandler.gradient);