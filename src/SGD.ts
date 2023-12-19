class SGD extends Optimiser{
    processGradient(gradient: Tensor, step: number, kwargs: any): Tensor {
        return new Tensor(Multiply.product([gradient.value,TensorUtils.filledArray(gradient.shape,kwargs.lr)]));
    }

}