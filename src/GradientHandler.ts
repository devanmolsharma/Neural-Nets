class GradientHandler {
    private backwardFunctions: ((gradient: Tensor) => Tensor)[] = [];
    private _children: Tensor[] = [];
    private _gradient: Tensor;
    constructor(private _tensor: Tensor) {
    }
    public get gradient(): Tensor {
        return this._gradient;
    }
    public set gradient(value: Tensor) {
        this._gradient = value;
    }
}