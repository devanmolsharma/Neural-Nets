class SGD extends Optimiser {
    private lr: number;
    private minLr: number;
    private decay: number;

    constructor(_parameters: Map<String, Tensor>[], { ...kwargs }) {
        super(_parameters, kwargs);
        this.lr = kwargs.lr ?? 1e-4;
        this.minLr = kwargs.lr * 1e-2;
        this.decay = kwargs.decay ?? 1e-2;
    };
    processGradient(gradient: Tensor, step: number, kwargs: any): Tensor {
        const unNaned = NanToNum.unNanify(gradient.value);
        const product = Multiply.product([unNaned as NumArray, TensorUtils.filledArray(gradient.shape, kwargs.lr)]);
                
        return new Tensor(product);
    }

    step(): void {
        super.step();
        this.lr = Math.max(this.minLr, this.lr - this.lr * this.decay)
    }

}