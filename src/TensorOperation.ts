
interface TensorOperation {
    forward(...tensors: Tensor[]): Tensor;
    backward(...gradients: Tensor[]): Tensor | Tensor[];
}
