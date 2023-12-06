// Abstract class representing a tensor operation
abstract class TensorOperation extends Function {
    // Bound instance of the TensorOperation to allow chaining
    private _bound: TensorOperation;

    // Constructor sets up the bound instance for chaining
    constructor() {
        // Call the super constructor to set up the function
        super('...args', 'return this._bound._call(...args)');

        // Bind the instance to enable chaining
        this._bound = this.bind(this);

        // Return the bound instance
        return this._bound;
    }

    // Verify shapes of tensors after the operation
    public verify(tensors: Tensor[]): void {
        // Call the _call method to perform the operation
        const out = this._call(tensors);

        // Get the gradients from the operation
        const grad = this.getGradient(out);

        // Check if the backward function returns gradients for all tensors
        if (grad.length != tensors.length) {
            throw "backward function does not return gradients of all tensors";
        }

        // Check if the shapes of gradients match the shapes of corresponding tensors
        for (const i in tensors) {
            if (JSON.stringify(tensors[i].shape) != JSON.stringify(grad[i].shape)) {
                throw `gradient shape mismatch on element at index ${i}.\n required shape: ${tensors[i].shape} but got shape ${grad[i].shape}`;
            }
        }

        // Log verification success
        console.log('Shapes verified for Class:' + this.constructor.name);
    }

    // Abstract method for forward pass of the operation
    public abstract forward(tensors: NumArray[], ...kwargs: any): NumArray;

    // Abstract method for backward pass of the operation
    public abstract backward(gradient: NumArray): NumArray[];

    // Abstract method for setting up the operation
    public abstract setup(tensors: Tensor[], ...kwargs: any): void;

    // Private method to perform the actual call to the operation
    private _call(tensors: Tensor[], ...kwargs: any): Tensor {
        // Set up the operation
        this.setup(tensors, ...kwargs);

        // Perform the forward pass and create a new tensor with the result
        const out = new Tensor(this.forward((tensors.map((t) => t.value)), ...kwargs));

        if (kwargs.withGrad ?? true) {

            // Register tensors as children and the operation for gradient tracking
            out.gradientHandler.registerChildren(tensors);
            out.gradientHandler.registerOperation(this);
        }
        // Return the resulting tensor
        return out;
    }

    // Get the gradient as tensors from a gradient tensor
    public getGradient(gradient: Tensor): Tensor[] {
        // Perform the backward pass and create new tensors for the gradients
        const grads = this.backward(gradient.value);
        return grads.map((e) => new Tensor(e));
    }
}
