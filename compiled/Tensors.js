// Class for managing gradients during backpropagation
class GradientHandler {
    // Constructor takes a tensor to associate with this handler
    constructor(_tensor) {
        this._tensor = _tensor;
        // Array to store child tensors
        this._children = [];
    }
    // Getter for child tensors
    get children() {
        return this._children;
    }
    // Getter for the gradient tensor
    get gradient() {
        // Throw an error if gradient is not set
        if (!this._gradient) {
            throw 'Tensor not connected to the main gradient tensor or backward() method not called';
        }
        return this._gradient;
    }
    // Setter for the gradient tensor
    set gradient(value) {
        this._gradient = value;
    }
    // Register child tensors with the handler
    registerChildren(children) {
        this._children.push(...children);
    }
    // Register the associated operation with the handler
    registerOperation(operation) {
        this.Operation = operation;
    }
    // Perform backward pass to compute gradients
    backward(previousGradient) {
        // Use provided gradient tensor or create a clone of the main tensor
        let temp = previousGradient;
        if (!previousGradient) {
            temp = this._tensor.clone();
        }
        // Check if the tensor is one-dimensional
        if (!previousGradient && JSON.stringify(temp.shape) !== JSON.stringify([1])) {
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
}
// Class representing a Tensor
class Tensor {
    // Constructor takes a NumArray as the initial value
    constructor(_value) {
        this._value = _value;
        // Gradient handler for backpropagation
        this.gradientHandler = new GradientHandler(this);
    }
    // Getter for the value, returning a deep clone to avoid unintentional modifications
    get value() {
        return structuredClone(this._value);
    }
    // Setter for the value
    set value(value) {
        this._value = value;
    }
    // Getter for the shape of the tensor
    get shape() {
        return this._calculateShape(this._value);
    }
    // Method to create a clone of the tensor
    clone() {
        return new Tensor(this.value);
    }
    // Private method to recursively calculate the shape of the tensor
    _calculateShape(array, shape = []) {
        shape.push(array.length);
        if (isNaN(array[0])) {
            this._calculateShape(array[0], shape);
        }
        return shape;
    }
    // Method for performing backpropagation
    backward(gradient) {
        this.gradientHandler.backward(gradient);
    }
    // Method to convert the tensor to a string representation
    toString() {
        return "Tensor: " + JSON.stringify(this.value) + " shape: " + this.shape;
    }
}
// Abstract class representing a tensor operation
class TensorOperation extends Function {
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
    verify(tensors) {
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
    // Private method to perform the actual call to the operation
    _call(tensors, ...kwargs) {
        var _a;
        // Set up the operation
        this.setup(tensors, ...kwargs);
        // Perform the forward pass and create a new tensor with the result
        const out = new Tensor(this.forward((tensors.map((t) => t.value)), ...kwargs));
        if ((_a = kwargs.withGrad) !== null && _a !== void 0 ? _a : true) {
            // Register tensors as children and the operation for gradient tracking
            out.gradientHandler.registerChildren(tensors);
            out.gradientHandler.registerOperation(this);
        }
        // Return the resulting tensor
        return out;
    }
    // Get the gradient as tensors from a gradient tensor
    getGradient(gradient) {
        // Perform the backward pass and create new tensors for the gradients
        const grads = this.backward(gradient.value);
        return grads.map((e) => new Tensor(e));
    }
}
// Class representing the sum operation for tensors
class Add extends TensorOperation {
    // Helper function to add arrays element-wise
    addArrays(a, b) {
        return a.map((v, i) => Array.isArray(v) ? this.addArrays(v, b[i]) : v + b[i]);
    }
    // Function to calculate the sum of arrays
    sum(arrays) {
        return arrays.reduce((a, b) => this.addArrays(a, b));
    }
    // Forward pass of the sum operation
    forward(tensors) {
        // Calculate the sum of the tensors
        const sum = this.sum(tensors);
        return sum;
    }
    // Backward pass of the sum operation
    backward(gradient) {
        // Distribute the gradient to all input tensors
        return Array(this.tensorCount).fill(gradient);
    }
    // Setup method to initialize the tensor count
    setup(tensors) {
        this.tensorCount = tensors.length;
    }
}
// Class representing the mean operation for tensors
class Mean extends TensorOperation {
    // Forward pass of the mean operation
    forward(tensors) {
        // Calculate the mean of the tensors
        const sum = tensors[0].flat().reduce((i, j) => i + j) / (this.elementCount);
        return [sum];
    }
    // Backward pass of the mean operation
    backward(gradient) {
        // Distribute the gradient equally to all elements in the tensors
        return [TensorUtils.filled(this.shape, gradient[0]).value];
    }
    // Setup method to initialize shape and element count
    setup(tensors) {
        this.shape = tensors[0].shape;
        this.elementCount = this.shape.reduce((x, y) => x * y);
    }
}
// Class representing the Matrix Multipication operation for tensors
class Matmul extends TensorOperation {
    static tensorMul(tensors) {
        // Calculate the mean of the tensors
        let finalArr = TensorUtils.filledArray([tensors[0].length, tensors[1][0].length], 0);
        for (let row = 0; row < finalArr.length; row++) {
            for (let col = 0; col < finalArr[0].length; col++) {
                let sum = 0;
                for (let i = 0; i < tensors[0][row].length; i++) {
                    sum += tensors[0][row][i] * tensors[1][i][col];
                }
                finalArr[row][col] = sum;
            }
        }
        return finalArr;
    }
    // Forward pass of the mean operation
    forward(tensors) {
        this.t1 = tensors[0];
        this.t2 = tensors[1];
        return Matmul.tensorMul([this.t1, this.t2]);
    }
    backward(gradient) {
        let grads = [Matmul.tensorMul([gradient, TensorUtils.transpose(this.t2)]), Matmul.tensorMul([this.t1, gradient])];
        return grads;
    }
    // Do Basic checks and store tensor shapes
    setup(tensors) {
        if (tensors.length != 2) {
            throw "this Operation requires exactly two Tensors";
        }
        const shape1 = tensors[0].shape;
        const shape2 = tensors[1].shape;
        if (shape1[1] != shape2[0]) {
            throw `Cannot multiply matrices with shapes ${shape1} and ${shape2}`;
        }
    }
}
// Max Operation for relu activation later
class Max extends TensorOperation {
    static max(tensor, limit) {
        if (isNaN(tensor)) {
            return tensor.map((val) => this.max(val, limit));
        }
        return tensor > limit ? tensor : limit;
    }
    static calcGrad(tensor, gradient, limit) {
        if (isNaN(tensor)) {
            return tensor.map((val, i) => this.calcGrad(val, gradient[i], limit));
        }
        return tensor > limit ? gradient : 0;
    }
    // Forward pass
    forward(tensors) {
        this.limit = tensors[1][0];
        this.tensor = tensors[0];
        return Max.max(tensors[0], tensors[1][0]);
    }
    backward(gradient) {
        let grads = [Max.calcGrad(this.tensor, gradient, this.limit), [0]];
        return grads;
    }
    // Do Basic checks
    setup(tensors) {
        if (tensors.length != 2) {
            throw "this Operation requires exactly two Tensors";
        }
        if (JSON.stringify(tensors[1].shape) != JSON.stringify([1])) {
            throw "tensor2 should have shape [1]";
        }
    }
}
// Factory class for creating tensors with different initial values and shapes
class TensorUtils {
    // Reshape a given array into the specified shape
    static reshape(array, shape) {
        let result = this.filledArray(shape);
        let flat = this.flatten(array);
        for (let i = 0; i < flat.length; i++) {
            let indices = this.getIndices(i, shape);
            this.setElement(result, indices, flat[i]);
        }
        return result;
    }
    // Transpose a given 2D array
    static transpose(array) {
        let result = this.filledArray([array[0].length, array.length]);
        for (let i = 0; i < array.length; i++) {
            for (let j = 0; j < array[i].length; j++) {
                result[j][i] = array[i][j];
            }
        }
        return result;
    }
    // Helper method to flatten an array
    static flatten(array) {
        return array.reduce((acc, val) => Array.isArray(val) ? acc.concat(this.flatten(val)) : acc.concat(val), []);
    }
    // Helper method to get multi-dimensional indices from a flat index
    static getIndices(index, shape) {
        let indices = [];
        for (let i = shape.length - 1; i >= 0; i--) {
            indices.unshift(index % shape[i]);
            index = Math.floor(index / shape[i]);
        }
        return indices;
    }
    // Helper method to set an element at the specified multi-dimensional indices
    static setElement(array, indices, value) {
        for (let i = 0; i < indices.length - 1; i++) {
            array = array[indices[i]];
        }
        array[indices[indices.length - 1]] = value;
    }
    // Create a filled array with the specified shape and fill value
    static filledArray(shape, fillValue = 0) {
        // If the shape is empty, return the fill value
        if (shape.length === 0) {
            return fillValue;
        }
        // Create a new array with the specified shape
        let array = new Array(shape[0]);
        // Recursively fill the array with nested arrays
        for (let i = 0; i < array.length; i++) {
            array[i] = this.filledArray(shape.slice(1), fillValue);
        }
        return array;
    }
    // Create a tensor with all elements set to 1 and the specified shape
    static ones(shape) {
        return new Tensor(this.filledArray(shape, 1));
    }
    // Create a tensor with all elements set to 0 and the specified shape
    static zeros(shape) {
        return new Tensor(this.filledArray(shape, 0));
    }
    // Create a tensor with all elements set to the specified fill value and shape
    static filled(shape, fillValue) {
        return new Tensor(this.filledArray(shape, fillValue));
    }
}
/// <reference path="TensorOperationsList.ts" />
const [t1, t2] = [new Tensor([[1, 2, 3], [1, 0, 0], [1, 0, 0]]), new Tensor([[1, -2], [3, -4], [-3, 4]])];
// let j = new Matmul()([t1, t2])
// j = new Max()([j,new Tensor([0])])
let mean = new Mean()([new Max()([t2, new Tensor([0])])]);
mean.backward();
console.log(t2.gradientHandler.gradient);
