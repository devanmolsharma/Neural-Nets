// Class for managing gradients during backpropagation
class GradientHandler {
    // Reference to the tensor associated with this gradient handler
    private Operation: TensorOperation;

    // Array to store child tensors
    private _children: Tensor[] = [];

    // The gradient tensor associated with the main tensor
    private _gradient: Tensor;

    // Constructor takes a tensor to associate with this handler
    constructor(private _tensor: Tensor) {
    }

    // Getter for child tensors
    public get children(): Tensor[] {
        return this._children;
    }

    // Getter for the gradient tensor
    public get gradient(): Tensor {
        // Throw an error if gradient is not set
        if (!this._gradient) {
            throw 'Tensor not connected to the main gradient tensor or backward() method not called';
        }
        return this._gradient;
    }

    // Setter for the gradient tensor
    public set gradient(value: Tensor) {
        this._gradient = value;
    }

    // Register child tensors with the handler
    public registerChildren(children: Tensor[]) {
        this._children.push(...children);
    }

    // Register the associated operation with the handler
    public registerOperation(operation: TensorOperation) {
        this.Operation = operation;
    }

    // Perform backward pass to compute gradients
    public backward(previousGradient?: Tensor) {
        // Use provided gradient tensor or create a clone of the main tensor
        let temp = previousGradient;
        if (!previousGradient) {
            temp = this._tensor.clone();
        }

        // Check if the tensor is one-dimensional
        if (!previousGradient && JSON.stringify(temp.shape!) !== JSON.stringify([1])) {
            throw 'cannot call backward() when the tensor is not one-dimensional';
        }

        // If an operation is registered, propagate the gradient to children
        if (this.Operation) {
            const childGrads = this.Operation.getGradient(temp);

            for (const i in this._children) {
                this._children[i].backward(childGrads[i]);
            }
        }

        // Set the gradient tensor for this handler
        this._gradient = temp;
    }


    applyGradient(){
        this._tensor.value =  Subtract.diff([this._tensor.value , this.gradient.value]);
    }
}
