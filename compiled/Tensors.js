class GradientHandler {
    constructor(_tensor) {
        this._tensor = _tensor;
        this._children = [];
    }
    get children() {
        return this._children;
    }
    get gradient() {
        if (!this._gradient) {
            throw 'Tensor not connected to the main gradient tensor or backward() method not called';
        }
        return this._gradient;
    }
    set gradient(value) {
        this._gradient = value;
    }
    registerChildren(children) {
        this._children.push(...children);
    }
    registerOperation(operation) {
        this.Operation = operation;
    }
    backward(previousGradient) {
        let temp = previousGradient;
        if (!previousGradient) {
            temp = this._tensor.clone();
        }
        if (!previousGradient && JSON.stringify(temp.shape) !== JSON.stringify([1])) {
            throw 'cannot call backward() when the tensor is not one dimentional';
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
class Tensor {
    constructor(_value) {
        this._value = _value;
        this.gradientHandler = new GradientHandler(this);
    }
    get value() {
        return structuredClone(this._value);
    }
    set value(value) {
        this._value = value;
    }
    get shape() {
        return this._calculateShape(this._value);
    }
    clone() {
        return new Tensor(this.value);
    }
    _calculateShape(array, shape = []) {
        shape.push(array.length);
        if (isNaN(array[0])) {
            this._calculateShape(array[0], shape);
        }
        return shape;
    }
    backward(gradient) {
        this.gradientHandler.backward(gradient);
    }
    toString() {
        return "Tensor: " + JSON.stringify(this.value) + " shape: " + this.shape;
    }
}
class TensorFactory {
    static filledArray(shape, fillValue = 0) {
        if (shape.length === 0) {
            return fillValue;
        }
        let array = new Array(shape[0]);
        for (let i = 0; i < array.length; i++) {
            array[i] = this.filledArray(shape.slice(1), fillValue);
        }
        return array;
    }
    static ones(shape) {
        return new Tensor(this.filledArray(shape, 1));
    }
    static zeros(shape) {
        return new Tensor(this.filledArray(shape, 0));
    }
    static filled(shape, fillValue) {
        return new Tensor(this.filledArray(shape, fillValue));
    }
}
class TensorOperation extends Function {
    constructor() {
        super('...args', 'return this._bound._call(...args)');
        this._bound = this.bind(this);
        return this._bound;
    }
    verify(tensors) {
        const out = this._call(tensors);
        const grad = this.getGradient(out);
        if (grad.length != tensors.length) {
            throw "backward function does not return gradients of all tensors";
        }
        for (const i in tensors) {
            if (JSON.stringify(tensors[i].shape) != JSON.stringify(grad[i].shape)) {
                throw `gradient shape mismatch on element at index ${i}.\n required shape: ${tensors[i].shape} but got shape ${grad[i].shape}`;
            }
        }
        console.log('Shapes verified for Class:' + this.constructor.name);
    }
    _call(tensors, ...kwargs) {
        this.setup(tensors, ...kwargs);
        const out = new Tensor(this.forward((tensors.map((t) => t.value)), ...kwargs));
        out.gradientHandler.registerChildren(tensors);
        out.gradientHandler.registerOperation(this);
        return out;
    }
    getGradient(gradient) {
        const grads = this.backward(gradient.value);
        return grads.map((e) => new Tensor(e));
    }
}
class Sum extends TensorOperation {
    addArrays(a, b) {
        return a.map((v, i) => Array.isArray(v) ? this.addArrays(v, b[i]) : v + b[i]);
    }
    sum(arrays) {
        return arrays.reduce((a, b) => this.addArrays(a, b));
    }
    forward(tensors) {
        const sum = this.sum(tensors);
        return sum;
    }
    backward(gradient) {
        return Array(this.tensorCount).fill(gradient);
    }
    setup(tensors) {
        this.tensorCount = tensors.length;
    }
}
class Mean extends TensorOperation {
    forward(tensors) {
        const sum = tensors.flat().reduce((i, j) => i + j) / (this.elementCount);
        return [sum];
    }
    backward(gradient) {
        return [TensorFactory.filled(this.shape, gradient[0]).value];
    }
    setup(tensors) {
        this.shape = tensors[0].shape;
        this.elementCount = this.shape.reduce((x, y) => x * y);
    }
}
/// <reference path="TensorOperationsList.ts" />
const [t1, t2, t3] = [new Tensor([1, 2, 3, 4]), new Tensor([1, 2, 3, 4]), new Tensor([1, 2, 3, 4])];
let j = new Sum()([t1, t2, t3]);
j = new Mean()([j]);
j.backward();
console.log(j.toString());
//# sourceMappingURL=Tensors.js.map