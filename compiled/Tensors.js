class Optimiser {
    constructor(_parameters, { ...kwargs }) {
        this._parameters = _parameters;
        this.stepNum = 0;
        this.extraArgs = kwargs;
    }
    ;
    get parameters() {
        return this._parameters;
    }
    step() {
        this._parameters.forEach((params) => {
            params.forEach((tensor) => {
                if (tensor instanceof Tensor) {
                    const processedGradient = this.processGradient(tensor.gradientHandler.gradient, this.stepNum, this.extraArgs);
                    tensor.gradientHandler.gradient = processedGradient;
                    tensor.gradientHandler.applyGradient();
                }
            });
        });
        this.stepNum++;
    }
    zero_grad() {
        this._parameters.forEach((params) => {
            params.forEach((tensor) => {
                if (tensor instanceof Tensor)
                    tensor.gradientHandler = new GradientHandler(tensor);
            });
        });
    }
}
/// <reference path="Optimiser.ts" />
class Adam extends Optimiser {
    constructor(_parameters, { ...kwargs }) {
        super(_parameters, kwargs);
    }
    ;
    processGradient(gradient, step, kwargs) {
        throw "Not Implemented Yet";
    }
    step() {
        super.step();
    }
}
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
    applyGradient() {
        this._tensor.value = Subtract.diff([this._tensor.value, this.gradient.value]);
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
/// <reference path="TensorOperation.ts" />
// Class representing a layer in neural network
class Layer {
    constructor() {
        // Parameters of the Layer
        this._parameters = new Map();
    }
    get parameters() {
        return this._parameters;
    }
    set parameters(value) {
        this._parameters = value;
    }
    registerParameter(name, parameter) {
        this._parameters.set(name, parameter);
    }
    isPrimitive(val) {
        if (val === Object(val)) {
            return false;
        }
        else {
            return true;
        }
    }
    // Planning on using this to save and load models
    toJson() {
        let j = [];
        (Object.getOwnPropertyNames(this)).forEach((name) => {
            const value = this[name];
            let tempValue;
            if (value instanceof Tensor) {
                tempValue = value.value;
            }
            else if (!this.isPrimitive(value)) {
                return;
            }
            else {
                tempValue = value;
            }
            j.push({
                'name': name,
                'value': tempValue
            });
        });
        return JSON.stringify({ layer: this.constructor.name, data: j });
    }
    loadData(json) {
        let params = json.data;
        params.forEach((data) => {
            if (data.value instanceof Array) {
                this[data.name] = new Tensor(data.value);
                this._parameters[data.name] = this[data.name];
            }
            else if (data.name) {
                this[data.name] = data.value;
                this._parameters[data.name] = this[data.name];
            }
        });
        return this;
    }
}
/// <reference path="Layer.ts" />
// a simple impelentation of Linear Layer
class Linear extends Layer {
    constructor(num_inputs = 1, num_out = 1, activation = null, useBias = true) {
        super();
        this.activation = activation;
        this.useBias = useBias;
        this.weights = TensorUtils.rand([num_out, num_inputs], -1e-3, 1e-3);
        if (useBias)
            this.biases = TensorUtils.rand([1, num_out], -1e-3, 1e-3);
        this.registerParameter('weights', this.weights);
        this.registerParameter('biases', this.biases);
        this.transposer = new Transpose();
        this.adder = new Add();
        this.matMul = new Matmul();
    }
    setupActivations() {
        if (this.activation && !this.activationFn)
            this.activationFn = eval(`new ${this.activation}`);
    }
    forward(input) {
        this.setupActivations();
        let x = this.matMul([input, this.transposer([this.weights])]);
        if (this.useBias)
            x = this.adder([x, this.biases]);
        if (this.activation)
            x = this.activationFn([x]);
        return x;
    }
}
class Metric {
}
class Accuracy extends Metric {
    constructor(afterIterations = 100) {
        super();
        this.afterIterations = afterIterations;
        this.corrects = 0;
    }
    forward(actual, pred, loop) {
        const accuracy = this.corrects / 100;
        this.reset(loop);
        if (TensorUtils.argmax(actual.value.flat(20)) == TensorUtils.argmax(pred.value.flat(20))) {
            this.corrects++;
        }
        return accuracy;
    }
    reset(loop) {
        if (loop % this.afterIterations == 0) {
            this.corrects = 0;
        }
    }
}
class Model extends Function {
    constructor() {
        // Call the super constructor to set up the function
        super('...args', 'return this._bound._call(...args)');
        // Bind the instance to enable chaining
        this._bound = this.bind(this);
        // Return the bound instance
        return this._bound;
    }
    _call(...params) {
        return this.forward(params);
    }
    isPrimitive(val) {
        if (val === Object(val)) {
            return false;
        }
        else {
            return true;
        }
    }
    toJson() {
        let j = [];
        (Object.getOwnPropertyNames(this)).forEach((name) => {
            const value = this[name];
            let tempValue;
            if (value instanceof Tensor) {
                tempValue = value.value;
            }
            else if (value instanceof Array && value[0] && value[0] instanceof Tensor) {
                tempValue = { array: value.map((e) => e.value) };
            }
            else if (value instanceof Layer) {
                tempValue = value.toJson();
            }
            else if (value instanceof Array && value[0] && value[0] instanceof Layer) {
                tempValue = { array: value.map((e) => e.toJson()) };
            }
            else if (!this.isPrimitive(value)) {
                return;
            }
            else {
                tempValue = value;
            }
            j.push({
                name: name,
                value: tempValue
            });
        });
        return j;
    }
    loadData(json) {
        let params = json;
        params.forEach((data) => {
            if (data.name === 'length')
                return;
            if (data.name === 'name')
                return;
            if (data.value instanceof Array && this.isPrimitive(data.value.flat[0])) {
                this[data.name] = new Tensor(data.value);
            }
            else if (data.value.layer) {
                const obj = eval(`new ${data.value.layer}()`);
                obj.loadData(data.value)(this)[data.name] = obj;
            }
            else if (data.value.array) {
                let temp = [];
                data.value.array.forEach((e) => {
                    if (!(e.value instanceof Array)) {
                        const jsn = JSON.parse(e);
                        const obj = eval(`new ${jsn.layer}()`);
                        obj.loadData(jsn);
                        temp.push(obj);
                    }
                    else {
                        const tensor = new Tensor(e);
                        temp.push(tensor);
                    }
                });
                this[data.name] = temp;
            }
            else {
                this[data.name] = data.value;
            }
        });
        return this;
    }
}
class SGD extends Optimiser {
    constructor(_parameters, { ...kwargs }) {
        var _a, _b;
        super(_parameters, kwargs);
        this.lr = (_a = kwargs.lr) !== null && _a !== void 0 ? _a : 1e-4;
        this.minLr = kwargs.lr * 1e-2;
        this.decay = (_b = kwargs.decay) !== null && _b !== void 0 ? _b : 1e-2;
    }
    ;
    processGradient(gradient, step, kwargs) {
        const unNaned = NanToNum.unNanify(gradient.value);
        const product = Multiply.product([unNaned, TensorUtils.filledArray(gradient.shape, kwargs.lr)]);
        return new Tensor(product);
    }
    step() {
        super.step();
        this.lr = Math.max(this.minLr, this.lr - this.lr * this.decay);
    }
}
/// <reference path="Model.ts" />
class Sequential extends Model {
    constructor() {
        super(...arguments);
        this._layers = [];
    }
    get layers() {
        return this._layers;
    }
    add(value) {
        this._layers.push(value);
    }
    forward(params) {
        let res;
        this._layers.forEach((layer, i) => {
            if (i == 0)
                res = layer.forward(params[0]);
            else
                res = layer.forward(res);
        });
        return res;
    }
    getParameters() {
        return this.layers.map((layer) => layer.parameters);
    }
}
/// <reference path="GradinetHandler.ts" />
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
        if (JSON.stringify(TensorUtils.calculateShape(value)) != JSON.stringify(this.shape))
            throw `shapes dont match , setting ${JSON.stringify(value)} to tensor ${JSON.stringify(this.value)} , shapes ${(TensorUtils.calculateShape(value))} , ${JSON.stringify(this.shape)} `;
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
    _calculateShape(array, shapeArr = []) {
        shapeArr.push(array.length);
        if (Array.isArray(array[0])) {
            this._calculateShape(array[0], shapeArr);
        }
        return shapeArr;
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
/// <reference path="TensorOperation.ts" />
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
        const sum = tensors[0].flat(20).reduce((i, j) => i + j) / (this.elementCount);
        return [sum];
    }
    // Backward pass of the mean operation
    backward(gradient) {
        // Distribute the gradient equally to all elements in the tensors
        return [TensorUtils.filledArray(this.shape, gradient[0])];
    }
    // Setup method to initialize shape and element count
    setup(tensors) {
        this.shape = tensors[0].shape;
        this.elementCount = this.shape.reduce((x, y) => x * y);
    }
}
class Transpose extends TensorOperation {
    // Forward pass of the Transpose operation
    forward(tensors) {
        // Calculate the Transpose of the tensors
        const transpose = TensorUtils.transpose(tensors[0]);
        return transpose;
    }
    // Backward pass of the Transpose operation
    backward(gradient) {
        // Distribute the gradient equally to all elements in the tensors
        const transpose = TensorUtils.transpose(gradient);
        return [transpose];
    }
    // Setup method to initialize shape and element count
    setup(tensors) {
    }
}
// Class representing the Matrix Multiplication operation for tensors
class Matmul extends TensorOperation {
    // Static method to perform matrix multiplication of two tensors
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
    // Forward pass of the Matmul operation
    forward(tensors) {
        this.t1 = tensors[0];
        this.t2 = tensors[1];
        return Matmul.tensorMul([this.t1, this.t2]);
    }
    // Backward pass of the Matmul operation
    backward(gradient) {
        let grads = [Matmul.tensorMul([gradient, TensorUtils.transpose(this.t2)]), Matmul.tensorMul([TensorUtils.transpose(this.t1), gradient])];
        return grads;
    }
    // Do basic checks and store tensor shapes
    setup(tensors) {
        if (tensors.length != 2) {
            throw "This operation requires exactly two tensors.";
        }
        const shape1 = tensors[0].shape;
        const shape2 = tensors[1].shape;
        if (shape1[1] != shape2[0]) {
            throw `Cannot multiply matrices with shapes ${shape1} and ${shape2}`;
        }
    }
}
// Class representing the Min operation for tensors
class Min extends TensorOperation {
    // Static method to find the minimum value between tensor and limit
    static min(tensor, limit) {
        if (Array.isArray(tensor)) {
            return tensor.map((val) => this.min(val, limit));
        }
        return tensor < limit ? tensor : limit;
    }
    static calcGrad(tensor, gradient, limit) {
        if (Array.isArray(tensor)) {
            return tensor.map((val, i) => this.calcGrad(val, gradient[i], limit));
        }
        return tensor < limit ? gradient : limit;
    }
    // Forward pass of the Min operation
    forward(tensors) {
        this.limit = tensors[1][0];
        this.tensor = tensors[0];
        return Min.min(tensors[0], tensors[1][0]);
    }
    // Backward pass of the Min operation
    backward(gradient) {
        let grads = [Min.calcGrad(this.tensor, gradient, this.limit), [0]];
        return grads;
    }
    // Do basic checks
    setup(tensors) {
        if (tensors.length != 2) {
            throw "This operation requires exactly two tensors.";
        }
        if (JSON.stringify(tensors[1].shape) != JSON.stringify([1])) {
            throw "Tensor 2 should have shape [1]";
        }
    }
}
// Max Operation for relu activation later
class Max extends TensorOperation {
    get limit() {
        return this._limit;
    }
    set limit(value) {
        this._limit = value;
    }
    // Static method to find the maximum value between tensor and limit
    static max(tensor, limit) {
        if (Array.isArray(tensor)) {
            return tensor.map((val) => this.max(val, limit));
        }
        return tensor > limit ? tensor : limit;
    }
    // Static method to calculate the gradient for the Max operation
    static calcGrad(tensor, gradient, limit) {
        if (Array.isArray(tensor)) {
            return tensor.map((val, i) => this.calcGrad(val, gradient[i], limit));
        }
        return tensor > limit ? gradient : 0;
    }
    // Forward pass of the Max operation
    forward(tensors) {
        this.tensor = tensors[0];
        return Max.max(tensors[0], this.limit);
    }
    // Backward pass of the Max operation
    backward(gradient) {
        let grads = [Max.calcGrad(this.tensor, gradient, this.limit), [0]];
        return grads;
    }
    // Do basic checks
    setup(tensors) {
        this.limit = tensors[1] ? tensors[1].value[0] : 0;
        if (tensors[1] && (JSON.stringify(tensors[1].shape) != JSON.stringify([1]))) {
            throw "Tensor 2 should have shape [1]";
        }
    }
}
// Max Operation for relu activation later
class Normalize extends TensorOperation {
    // Static method to find the maximum value between tensor and limit
    static norm(tensor, max, mean, variance, stdiv) {
        if (Array.isArray(tensor)) {
            return tensor.map((val) => this.norm(val, max, mean, variance, stdiv));
        }
        return (tensor - mean) / stdiv;
    }
    // Forward pass of the Max operation
    forward(tensors) {
        const flatt0 = tensors[0].flat(20);
        this.max = Math.max(flatt0.reduce((x, y) => (x > y) ? x : y), 1);
        this.mean = flatt0.reduce((x, y) => x + y) / flatt0.length;
        this.variance = flatt0.reduce((sum, val) => sum + Math.pow(val - this.mean, 2), 0) / flatt0.length;
        this.stdDev = Math.sqrt(this.variance);
        return Normalize.norm(tensors[0], this.max, this.mean, this.variance, this.stdDev);
    }
    // Static method to calculate the gradient for the Max operation
    static calcGrad(gradient, max, mean, variance, stdiv) {
        if (Array.isArray(gradient)) {
            return gradient.map((val, i) => this.calcGrad(val, max, mean, variance, stdiv));
        }
        return (gradient / stdiv);
    }
    // Backward pass of the Max operation
    backward(gradient) {
        let grads = [Normalize.calcGrad(gradient, this.max, this.mean, this.variance, this.stdDev)];
        return grads;
    }
    // Do basic checks
    setup(tensors) {
    }
}
class Rescale extends TensorOperation {
    // Static method to find the maximum value between tensor and limit
    static rescale(tensor, max) {
        if (Array.isArray(tensor)) {
            return tensor.map((val) => this.rescale(val, max));
        }
        return tensor / max;
    }
    // Forward pass of the Max operation
    forward(tensors) {
        const flatt0 = tensors[0].flat(20);
        this.max = Math.max(flatt0.reduce((x, y) => (x > y) ? x : y), 1);
        return Rescale.rescale(tensors[0], this.max);
    }
    // Static method to calculate the gradient for the Max operation
    static calcGrad(gradient, max) {
        if (Array.isArray(gradient)) {
            return gradient.map((val, i) => this.calcGrad(val, max));
        }
        return (gradient / max);
    }
    // Backward pass of the Max operation
    backward(gradient) {
        let grads = [Rescale.calcGrad(gradient, this.max)];
        return grads;
    }
    // Do basic checks
    setup(tensors) {
    }
}
class NanToNum extends TensorOperation {
    // Static method to find the maximum value between tensor and limit
    static unNanify(tensor) {
        if (Array.isArray(tensor)) {
            return tensor.map((val) => this.unNanify(val));
        }
        if (isNaN(tensor) || tensor == null)
            return 0;
        return tensor;
    }
    // Forward pass of the Max operation
    forward(tensors) {
        this.tensor = tensors[0];
        return NanToNum.unNanify(tensors[0]);
    }
    // Backward pass of the Max operation
    backward(gradient) {
        let grads = [NanToNum.calcGrad(this.tensor, gradient), [0]];
        return grads;
    }
    static calcGrad(tensor, gradient) {
        if (Array.isArray(tensor)) {
            return tensor.map((val, i) => this.calcGrad(val, gradient[i]));
        }
        if (isNaN(tensor))
            return 0;
        return 1;
    }
    // Do basic checks
    setup(tensors) {
    }
}
// Class representing the Min operation for tensors
class Sigmoid extends TensorOperation {
    // Static method to find the minimum value between tensor and limit
    static sig(tensor) {
        if (Array.isArray(tensor)) {
            return tensor.map((val) => this.sig(val));
        }
        return 1 / (1 + Math.exp(-tensor));
    }
    // Forward pass of the Min operation
    forward(tensors) {
        this.result = Sigmoid.sig(tensors[0]);
        return structuredClone(this.result);
    }
    // Backward pass of the Min operation
    backward(gradient) {
        const res = [Multiply.product([
                gradient,
                Multiply.product([
                    structuredClone(this.result), Subtract.diff([TensorUtils.filledArray(this.shape, 1), structuredClone(this.result)])
                ])
            ])
        ];
        return res;
    }
    // Do basic checks
    setup(tensors) {
        this.shape = tensors[0].shape;
    }
}
class Softmax extends TensorOperation {
    // Static method to find the minimum value between tensor and limit
    static softmax(tensor, sum) {
        if (Array.isArray(tensor)) {
            return tensor.map((val) => this.softmax(val, sum));
        }
        return Math.exp(tensor) / sum;
    }
    // Forward pass of the Min operation
    forward(tensors) {
        this.sum = tensors[0].flat(20).reduce((prev, curr) => {
            return prev + Math.exp(curr);
        });
        this.result = Softmax.softmax(tensors[0], this.sum);
        return structuredClone(this.result);
    }
    // Backward pass of the Min operation
    backward(gradient) {
        let grad = TensorUtils.filledArray([this.shape[1], this.shape[1]], 0);
        for (let x = 0; x < this.shape[1]; x++) {
            for (let y = 0; y < this.shape[1]; y++) {
                if (x == y) {
                    const val = this.result.flat(20)[x];
                    grad[x][y] = val * (1 - val);
                }
                else {
                    const val1 = this.result.flat(20)[x];
                    const val2 = this.result.flat(20)[y];
                    grad[x][y] = -1 * val1 * val2;
                }
            }
        }
        return [Matmul.tensorMul([gradient, grad])];
    }
    // Do basic checks
    setup(tensors) {
        this.shape = tensors[0].shape;
        tensors[0] = new Rescale()([tensors[0]]);
    }
}
class ReLU extends Max {
}
class LeakyReLU extends Max {
    setup(tensors) {
        super.limit = -0.01;
    }
}
// Class representing the Subtraction operation for tensors
class Subtract extends TensorOperation {
    // Helper function to subtract arrays element-wise
    static subtract2DArrays(a, b) {
        return a.map((v, i) => Array.isArray(v) ? this.subtract2DArrays(v, b[i]) : v - b[i]);
    }
    // Function to calculate the difference of arrays
    static diff(arrays) {
        return arrays.reduce((a, b) => Subtract.subtract2DArrays(a, b));
    }
    // Forward pass of the Subtract operation
    forward(tensors) {
        // Calculate the difference of the tensors
        const diff = Subtract.diff(tensors);
        return diff;
    }
    // Backward pass of the Subtract operation
    backward(gradient) {
        // Distribute the gradient to all input tensors
        return [gradient, TensorUtils.reshape(gradient.flat(20).map((i) => -i), TensorUtils.calculateShape(gradient))];
    }
    // Setup method to initialize the tensor count
    setup(tensors) {
        if (tensors.length != 2) {
            throw "Subtraction is only valid operation for two tensors.";
        }
    }
}
class CrossEntropy extends TensorOperation {
    // Helper function to subtract arrays element-wise
    static crossEntropy2DArrays(a, b) {
        return a.map((v, i) => Array.isArray(v) ? this.crossEntropy2DArrays(v, b[i]) : (-Math.log(1 + 1e-8 - (Math.min(v, 1) - b[i]) ** 2)));
    }
    static crossEntropyBack(a, b) {
        return a.map((v, i) => Array.isArray(v) ? this.crossEntropyBack(v, b[i]) : ((2 * (v - b[i])) / (1 + 1e-8 - ((v - b[i]) ** 2))));
    }
    // Function to calculate the difference of arrays
    static ce(arrays) {
        return arrays.reduce((a, b) => CrossEntropy.crossEntropy2DArrays(a, b));
    }
    // Forward pass of the Subtract operation
    forward(tensors) {
        // Calculate the difference of the tensors
        this.tensors = tensors;
        return CrossEntropy.ce(tensors);
    }
    // Backward pass of the Subtract operation
    backward(gradient) {
        // Distribute the gradient to all input tensors
        return [CrossEntropy.crossEntropyBack(this.tensors[0], this.tensors[1]), gradient];
    }
    // Setup method to initialize the tensor count
    setup(tensors) {
        if (tensors.length != 2) {
            throw "Subtraction is only valid operation for two tensors.";
        }
    }
}
// Class representing the Multiply operation for tensors
class Multiply extends TensorOperation {
    // Helper function to multiply arrays element-wise
    static multiplyArrays(a, b) {
        return a.map((v, i) => Array.isArray(v) ? this.multiplyArrays(v, b[i]) : v * b[i]);
    }
    // Function to calculate the product of arrays
    static product(arrays) {
        return arrays.reduce((a, b) => this.multiplyArrays(a, b));
    }
    // Forward pass of the Multiply operation
    forward(tensors) {
        // Calculate the product of the tensors
        const res = Multiply.product(tensors);
        return res;
    }
    // Backward pass of the Multiply operation
    backward(gradient) {
        // Distribute the gradient to all input tensors
        if (this.tensors.length == 2) {
            return [Multiply.product([this.tensors[1], gradient]), Multiply.product([this.tensors[0], gradient])];
        }
        else {
            let grads = [];
            for (const array of this.tensors) {
                grads.push(Multiply.product([...this.tensors.filter((v) => v != array), gradient]));
            }
            return grads;
        }
    }
    // Setup method to initialize the tensors
    setup(tensors) {
        if (tensors.length < 2) {
            throw "Multiplication is only valid for over two tensors.";
        }
        ;
        this.tensors = tensors.map((e) => e.value);
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
    static argmax(arr) {
        return (arr).findIndex((v) => v == Math.max(...arr));
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
    static randArray(shape, min = 0, max = 1) {
        // If the shape is empty, return the fill value
        if (shape.length === 0) {
            return (Math.random() * (max - min)) + min;
        }
        // Create a new array with the specified shape
        let array = new Array(shape[0]);
        // Recursively fill the array with nested arrays
        for (let i = 0; i < array.length; i++) {
            array[i] = this.randArray(shape.slice(1), min, max);
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
    // Create a tensor with all elements set to the specified fill value and shape
    static rand(shape, min, max) {
        return new Tensor(this.randArray(shape, min, max));
    }
    static calculateShape(array, shape = []) {
        shape.push(array.length);
        if (Array.isArray(array[0])) {
            TensorUtils.calculateShape(array[0], shape);
        }
        return shape;
    }
}
class Trainer {
    constructor(net, optimiser, loss, metric = null) {
        this.net = net;
        this.optimiser = optimiser;
        this.loss = loss;
        this.metric = metric;
    }
    onTrainingDone(net) { }
    onLoopDone(loss, expected, out, loopNo, metric) { }
    loop(x, y, loopNo) {
        let out = this.net.forward([new Tensor(x)]);
        const expected = new Tensor(y);
        let loss = this.loss([out, expected]);
        loss = new Mean()([loss]);
        loss.backward();
        this.optimiser.step();
        this.optimiser.zero_grad();
        if (loss.value == Infinity) {
            throw 'Infinity Seen';
        }
        this.onLoopDone(loss, expected, out, loopNo, this.calculateMetric(expected, out, loopNo));
    }
    calculateMetric(expected, out, loopNo) {
        return this.metric.forward(expected, out, loopNo);
    }
    train(x, y, oneHotEncode = false) {
        for (let i = 0; i < x.length; i++) {
            const inp = x[i];
            let exp = y[i];
            if (oneHotEncode) {
                let oneHot = new Array(10).fill(0);
                oneHot[exp] = 0.8;
                exp = [oneHot];
            }
            setTimeout(((inp, exp, i) => {
                return () => this.loop(inp, exp, i);
            })(inp, exp, i), 0);
        }
        this.onTrainingDone(this.net);
    }
}
function convertToModelCode(config) {
    const modelCode = [];
    // Create model
    modelCode.push('var model = new Sequential();');
    // Add layers
    config.layers.forEach((layer) => {
        modelCode.push(`model.add(new Linear(${layer.inputs},${layer.outputs},'${layer.activation}'));`);
    });
    // Create optimizer
    const optimizerCode = `let opt = new SGD(model.getParameters(), {lr:${config.optimizer.learningRate || 1e-2}, decay:${config.optimizer.decay || 0.2}, minLr:${config.optimizer.minLr || 1e-4}});`;
    modelCode.push(optimizerCode);
    // Create loss
    const lossCode = `let lossFn = new ${config.loss}();`;
    modelCode.push(lossCode);
    // Create metric
    const metricCode = `const metric = new ${config.metric}(${config.metric === 'Accuracy' ? 100 : ''});`;
    modelCode.push(metricCode);
    // Create trainer
    modelCode.push('var trainer = new Trainer(model, opt, lossFn, metric);');
    return modelCode.join('\n');
}
async function loadModelFromJson(filePath) {
    // var model;
    const res = await fetch(filePath);
    const parsedJson = await res.json();
    const config = parsedJson[0];
    const weights = parsedJson[1];
    const modelCode = [];
    // Create model
    modelCode.push('var model = new Sequential();');
    // Add layers
    config.layers.forEach((layer) => {
        modelCode.push(`model.add(new Linear(${layer.inputs},${layer.outputs},'${layer.activation}'));`);
    });
    const code = modelCode.join('\n');
    eval(code);
    // @ts-ignore
    model.loadData(weights);
    // @ts-ignore
    return model;
}
async function loadModelData(net, jsonPath) {
    net.loadData(await (await fetch(jsonPath)).text());
}
async function parseData(file) {
    const reader = new FileReader();
    reader.readAsText(file);
    let data = [];
    return await new Promise((resolve, reject) => {
        reader.onloadend = () => {
            const lines = reader.result.split('\n');
            lines.forEach((line) => {
                data.push([...line.split(',').map((n) => eval(n))]);
            });
            return resolve(data);
        };
    });
}
let draw = false;
let arr = TensorUtils.filledArray([28, 28], 0);
var model;
async function load() {
    model = await loadModelFromJson('./mnist.json');
    console.log('====================================');
    console.log(model);
    console.log('====================================');
}
let brushSize = 1;
document.addEventListener('DOMContentLoaded', () => {
    load();
    let table1 = document.getElementById('mnistVisualiser');
    for (let i = 0; i < 28; i++) {
        let tr = document.createElement('tr');
        for (let j = 0; j < 28; j++) {
            tr.innerHTML += '<td></td>';
        }
        table1.appendChild(tr);
    }
    let table = document.getElementById('drawBase');
    if (table) {
        for (let i = 0; i < 28; i++) {
            let tr = document.createElement('tr');
            for (let j = 0; j < 28; j++) {
                tr.innerHTML += `<td class="pixel ${i}-${j}"></td>`;
            }
            table.appendChild(tr);
        }
        const pixels = document.getElementsByClassName('pixel');
        document.getElementById('clear').addEventListener('click', () => {
            arr = TensorUtils.filledArray([28, 28], 0);
            for (const pixel of pixels) {
                pixel.classList.contains('black') && pixel.classList.remove('black');
            }
        });
        for (const pixel of pixels) {
            pixel.addEventListener('pointermove', (e) => {
                if (draw) {
                    // Use the brushSize to update adjacent pixels
                    const [x, y] = (e.target).classList[1].split('-').map(i => parseInt(i));
                    for (let i = x - brushSize + 1; i <= x + brushSize - 1; i++) {
                        for (let j = y - brushSize + 1; j <= y + brushSize - 1; j++) {
                            if (i >= 0 && i < 28 && j >= 0 && j < 28) {
                                const targetPixel = document.getElementsByClassName(`${i}-${j}`)[0];
                                if (!targetPixel.classList.contains('black')) {
                                    targetPixel.classList.add('black');
                                    arr[i][j] = 1;
                                }
                            }
                        }
                        // Update the display
                        let display = document.getElementById('mnistVisualiser');
                        for (let i = 1; i < 28; i++) {
                            for (let j = 0; j < 28; j++) {
                                const value = arr[i][j] * 255;
                                (display.children[i].children[j]).style.backgroundColor = `rgb(${value},${value},${value})`;
                            }
                        }
                        const pred = model.forward([new Tensor([arr.flat()])]).value.flat();
                        const highest = TensorUtils.argmax(pred);
                        console.log(model.layers[1]);
                        console.log('====================================');
                        console.log(pred);
                        console.log('====================================');
                        document.getElementById("prediction").innerHTML = `
                                    <h2 class="mb-4">Prediction</h2>
                                    <div class="alert alert-info" role="alert">
                                        It seems to be <strong>${highest}</strong>
                                    </div>
                                    <div class="mt-4">
                                        ${Object.entries(pred).map(([digit, probability]) => `
                                            <div class="progress mb-3">
                                                <div class="progress-bar" role="progressbar" style="width: ${probability * 100}%;"
                                                    aria-valuenow="${probability * 100}" aria-valuemin="0" aria-valuemax="100">
                                                    ${digit}
                                                </div>
                                            </div>
                                        `).join('')}
                                    </div>
`;
                        // 'I guess it is '+ argmax(model.forward([new Tensor([arr.flat()])]).value.flat() as any as number[]);
                    }
                }
            });
            pixel.addEventListener('pointerdown', (e) => {
                draw = true;
            });
            window.addEventListener('pointerup', (e) => {
                draw = false;
            });
        }
    }
});
