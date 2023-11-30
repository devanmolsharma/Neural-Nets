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
    registerChildren(...children) {
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
        console.log(temp.toString());
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
class TensorOperation extends Function {
    constructor() {
        super('...args', 'return this._bound._call(...args)');
        this._bound = this.bind(this);
        return this._bound;
    }
    verify(...tensors) {
        const out = this._call(...tensors);
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
    _call(...tensors) {
        const out = new Tensor(this.forward(...(tensors.map((t) => t.value))));
        out.gradientHandler.registerChildren(...tensors);
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
    forward(...tensors) {
        const sum = this.sum(tensors);
        return sum;
    }
    backward(gradient) {
        return [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]];
    }
}
class Mean extends TensorOperation {
    forward(...tensors) {
        const sum = tensors.flat().reduce((i, j) => i + j);
        return [sum];
    }
    backward(gradient) {
        return [[1]];
    }
}
const [t1, t2, t3] = [new Tensor([1, 2, 3, 4]), new Tensor([1, 2, 3, 4]), new Tensor([1, 2, 3, 4])];
let j = new Sum()(t1, t2, t3);
j = new Mean()(j);
j.backward();
console.log(j.gradientHandler.gradient.toString());
//# sourceMappingURL=Tensors.js.map