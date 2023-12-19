abstract class Optimiser {
    private extraArgs: any;
    private stepNum: number = 0;

    constructor(private _parameters: Map<String, Tensor>[], { ...kwargs }) {
        this.extraArgs = kwargs;
    };

    public get parameters(): Map<String, Tensor>[] {
        return this._parameters;
    }

    abstract processGradient(gradient: Tensor, step: number, kwargs: any): Tensor;

    step(): void {
        this._parameters.forEach((params: Map<String, Tensor>) => {
            params.forEach((tensor) => {
                const processedGradient = this.processGradient(tensor.gradientHandler.gradient, this.stepNum, this.extraArgs);
                tensor.gradientHandler.gradient = processedGradient;
                tensor.gradientHandler.applyGradient();
            })

        })
        this.stepNum++;
    }

    zero_grad() {
        this._parameters.forEach((params: Map<String, Tensor>) => {
            params.forEach((tensor) => {
                tensor.gradientHandler = new GradientHandler(tensor);
            })

        });
    }
}