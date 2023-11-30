class GradientHandler {
    private Operation: TensorOperation;
    private _children: Tensor[] = [];
    private _gradient: Tensor;

    constructor(private _tensor: Tensor) {
    }

    public get children(): Tensor[] {
        return this._children;
    }

    public get gradient(): Tensor {
        if (!this._gradient) {
            throw 'Tensor not connected to the main gradient tensor or backward() method not called';
        }
        return this._gradient;
    }

    public set gradient(value: Tensor) {
        this._gradient = value;
    }

    public registerChildren(children: Tensor[]) {
        this._children.push(...children);
    }

    public registerOperation(operation: TensorOperation) {
        this.Operation = operation;
    }

    public backward(previousGradient?: Tensor) {
        let temp = previousGradient;
        
        if (!previousGradient) {
            temp = this._tensor.clone();
        }
        if (!previousGradient && JSON.stringify(temp.shape!) !== JSON.stringify([1])) {
            throw 'cannot call backward() when the tensor is not one dimentional'
        }
        
        if (this.Operation) {
            const childGrads = this.Operation.getGradient(temp);

            for (const i in this._children) {
                this._children[i].backward(childGrads[i]);
            }

        }
        this._gradient = temp;
    }
}