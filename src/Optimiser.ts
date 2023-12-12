abstract class Optimiser {
    private extraArgs: any;
    private stepNum: number = 0;

    constructor(private _tensors: Tensor[], { ...kwargs }) {
        this.extraArgs = kwargs;
    };

    public get tensors(): Tensor[] {
        return this._tensors;
    }

    abstract processGradient(gradient: Tensor, step: number, kwargs:any ): Tensor;

    step(): void {
        this.tensors.forEach((tensor) => {
            const processedGradient = this.processGradient(tensor.gradientHandler.gradient, this.stepNum, this.extraArgs);
            tensor.gradientHandler.gradient = processedGradient;
        })
        this.stepNum++;
    }

    zero_grad() {
        this.tensors.forEach((tensor) => {
            tensor.gradientHandler = new GradientHandler(tensor);
        })
    }
}