class Tensor {
    public value: Array<any>;
    public shape: number[];
    public require_grad: boolean;
    public gradientFunctions: Function[];
    public children: Tensor[];
    public backFunctions: Function[];
    public parent_grads: any[];

    constructor(
        value: Array<any>,
        shape: number[] = value.length > 0 ? Tensor.calculateShape(value) : [0],
        require_grad: boolean = true,
        gradientFunctions: Function[] = [],
        children: Tensor[] = [],
        backFunctions: Function[] = [],
        parent_grads: any[] = []
    ) {
        if (!value) throw new Error("value not defined");
        this.value = value;
        this.shape = shape;
        this.require_grad = require_grad;
        this.gradientFunctions = gradientFunctions;
        this.children = children;
        this.backFunctions = backFunctions;
        this.parent_grads = parent_grads;
    }

    registerChildren(children: Tensor[]) {
        for (const child of children) {
            this.children.push(child);
        }
    }

    static calculateShape(array: any[], shape: number[] = []): number[] {
        shape.push(array.length);
        if (isNaN(array[0])) {
            Tensor.calculateShape(array[0], shape);
        }
        return shape;
    }

    addGradientFunction(fun: Function, fun2: Function, tensor2: Tensor): void {
        if (this.require_grad) {
            if (fun) {
                let copy = this.copy();
                copy.require_grad = false;
                this.gradientFunctions.push(() => copy.operation(tensor2, fun, false));
                if (fun2) {
                    if (tensor2 instanceof Tensor) {
                        tensor2.gradientFunctions.push(() => tensor2.operation(this, fun2, false));
                    }
                }
            }
        }
    }

    static onesLike(shapeArr: number[], base = true): Tensor | number[] {
        if (!base && shapeArr.length == 1) return new Array(shapeArr[0]).fill(1);
        else if (shapeArr.length == 1) return new Tensor(new Array(shapeArr[0]).fill(1));
        let arr = new Array(shapeArr[0]).fill(Tensor.onesLike(shapeArr.slice(1)), false);
        return base ? new Tensor(arr) : arr;
    }

    resetGrads(): void {
        this.gradients = null;
        this.parent_grads = [];
        this.gradientFunctions = [];
    }

    backward(): void {
        let gradients = Tensor.onesLike(this.shape);

        if (this.parent_grads.length > 0) {
            gradients = Tensor.multiply(this.parent_grads.reduce((x, y) => Tensor.add(x, y)), gradients);
        }

        for (let i = 0; i < this.gradientFunctions.length; i++) {
            if (gradients.length == 1) {
                gradients = this.gradientFunctions[i]();
            } else {
                gradients.multiply(this.gradientFunctions[i]());
            }
        }

        for (let i = 0; i < this.children.length; i++) {
            let grads = gradients.copy();
            this.children[i].parent_grads.push(this.backFunctions[i](grads));
            if (this.children[i].children.length > 0)
                this.children[i].backward(this.backFunctions[i](grads));
        }
    }

    getGradients(): Tensor {
        let gradients;
        if (!gradients && this.parent_grads !== []) {
            gradients = this.parent_grads.length > 1 ?
                this.parent_grads.reduce((x, y) => Tensor.add(x, y)) : this.parent_grads[0];
        }

        for (let i = 0; i < this.gradientFunctions.length; i++) {
            if (!gradients) {
                gradients = this.gradientFunctions[i]();
            } else {
                gradients = gradients.multiply(this.gradientFunctions[i](gradients));
            }

        }
        this.gradients = gradients;
        return gradients;
    }

    clearGradients(): void {
        this.gradientFunctions = [];
    }

    doOperation(func: Function, addArr: Array<any> | null, rem_shape?: Array<any> | null, rem_arr?: Array<any> | null): Array<any> {
        let newArr = [];

        if (!rem_shape) {
            rem_shape = this.shape;
            rem_arr = this.value;
        }

        if (rem_shape.length > 1) {
            for (let i = 0; i < rem_shape[0]; i++) {
                addArr ? newArr.push(this.doOperation(func, addArr[i], rem_shape.slice(1), rem_arr[i])) :
                    newArr.push(this.doOperation(func, null, rem_shape.slice(1), rem_arr[i]));
            }
            return newArr;
        }

        if (rem_arr != null) {
            for (let index = 0; index < rem_arr.length; index++) {
                newArr.push(func(rem_arr[index], addArr ? addArr[index] : null));
            }
        }
        return newArr;
    }

    operation(tensor2: Tensor | number, tensorFun: Function, returnSelf = true
        , bf1?: Function, bf2?: Function): Tensor {

        if (tensor2 instanceof Tensor) {
            if (this.shape.toString() !== tensor2.shape.toString()) {
                throw "Shapes do not match";
            }

            if (returnSelf) {
                this.value = this.doOperation((val, val2) => tensorFun(val, val2), tensor2.value);
                return this;
            } else {
                return new Tensor(
                    this.doOperation((val, val2) => tensorFun(val, val2), tensor2.value),
                    this.require_grad,
                    [this, tensor2],
                    [bf1, bf2]
                );
            }
        } else if (!isNaN(tensor2)) {
            if (returnSelf) {
                this.value = this.doOperation((val, _) => tensorFun(val, tensor2));
                return this;
            } else {
                return new Tensor(
                    this.doOperation((val, _) => tensorFun(val, tensor2)),
                    this.require_grad,
                    [this],
                    [bf1]
                );
            }
        }

        return this;
    }

    add(tensor2: Tensor | number) {
        this.operation(tensor2, (x, y) => x + y);

        if (!(tensor2 === this)) {
            this.addGradientFunction((x, y) => 1, (x, y) => 1, tensor2);
        } else {
            this.addGradientFunction((x, y) => 2, (x, y) => 1, tensor2);
        }

        return this;
    }

    static add(tensor1: Tensor, tensor2: Tensor | number): Tensor {
        return tensor1.operation(tensor2, (x: number, y: number) => x + y, false, (e: number) => e, (e: number) => e);
    }

    multiply(tensor2: Tensor | number): Tensor {
        if (!(tensor2 === this)) {
            if ((tensor2 instanceof Tensor) && tensor2.shape.flat() == 1) {
                tensor2 = tensor2.value.flat()[0];
            }

            if ((tensor2 instanceof Tensor) && this.shape.flat() == 1) {
                let value = Tensor.multiply(tensor2.copy(), this.value.flat()[0]).value;
                this.value = value;
                this.shape = tensor2.shape;
                this.gradientFunctions.push(() => new Tensor([value.reduce((x, y) => (x + y) / 2)]));
                return this;
            }

            this.addGradientFunction((_, y) => y, (_, y) => y, tensor2 instanceof Tensor ? tensor2.copy() : tensor2);

        } else {

            this.addGradientFunction((_, y) => 2 * y, (_, __) => 0, this.copy());
        }

        this.operation(tensor2 instanceof Tensor ? tensor2.copy() : tensor2, (x, y) => x * y);

        return this;
    }

    static multiply(tensor1: Tensor, tensor2: Tensor): Tensor {
        return tensor1.operation(
            tensor2,
            (x, y) => x * y,
            false,
            (e) => e.multiply(tensor2 instanceof Tensor ? tensor2.copy() : tensor2),
            (e) => e.multiply(tensor1.copy())
        );
    }

    divide(tensor2) {
        if (!(tensor2 === this)) {
            this.addGradientFunction((x, y) => 1 / y, (x, y) => 1 / y, tensor2);
        } else {
            this.addGradientFunction((x, y) => 0, (x, y) => 1, tensor2.copy());
        }

        return this.operation(tensor2, (x, y) => {
            if (y === 0) {
                throw "Zero value encountered";
            }
            return x / y;
        });
    }

    static divide(tensor1, tensor2) {
        return tensor1.operation(
            tensor2,
            (x, y) => {
                if (y === 0) {
                    throw "Zero value encountered";
                }
                return x / y;
            },
            false,
            (e) => Tensor.divide(e, tensor2 instanceof Tensor ? tensor2.copy() : tensor2),
            (e) => Tensor.divide(Tensor.onesLike(tensor2.shape).multiply(-1).multiply(tensor1.copy()), e.multiply(e))
        );
    }

    reshape(newShape) {
        if (this.require_grad) {
            let oldshape = this.shape;
            this.addGradientFunction((grad) => grad.reshape(oldshape));
        }

        this.value = this.reshapeValue(this.value, newShape);
        this.shape = newShape;

        return this;
    }

    reshapeValue(value, newShape) {
        if (newShape.length === 1) {
            return value;
        }

        const [dim, ...restShape] = newShape;
        const result = [];
        const step = Math.floor(value.length / dim);

        for (let i = 0; i < dim; i++) {
            result.push(this.reshapeValue(value.slice(i * step, (i + 1) * step), restShape));
        }

        return result;
    }

    repeat(repeats) {
        if (this.require_grad) {
            this.gradientFunctions.push((grad) => Tensor.gradRepeat(grad, this.shape, repeats));
        }

        this.value = this.repeatValue(this.value, repeats);
        this.shape = Tensor.calculateShape(this.value);

        return this;
    }

    repeatValue(value, repeats) {
        if (repeats.length === 0) {
            return value;
        }

        const [repeat, ...restRepeats] = repeats;
        const result = [];

        for (let i = 0; i < repeat; i++) {
            result.push(this.repeatValue(value, restRepeats));
        }

        return result;
    }

    subtract(tensor2) {
        this.addGradientFunction((x, y) => 1, (x, y) => -1, tensor2);
        return this.operation(tensor2, (x, y) => x - y);
    }

    static subtract(tensor1, tensor2) {
        return tensor1.operation(tensor2, (x, y) => x - y, false, e => e, e => e.multiply(-1));
    }

    static powerE(tensor1) {
        let tensor2 = 2.718281828459045235360;
        return tensor1.operation(tensor2, (x, y) => y ** x, false, e => e ** tensor2, e => 0);
    }

    powerE() {
        let tensor2 = 2.718281828459045235360;
        this.operation(tensor2, (x, y) => y ** x);
        this.addGradientFunction((x, y) => y, (x, y) => 0, this.copy());
        return this;
    }

    static sumAll(tensor1) {
        let value = tensor1.value.reduce((x, y) => x + y);
        return new Tensor([value], true, [tensor1], [() => Tensor.onesLike(Array.from(tensor1.shape))]);
    }

    sumAll() {
        let shape = Array.from(this.shape);
        this.gradientFunctions.push(() => Tensor.onesLike(shape));
        this.value = [this.value.flat(this.shape.length).reduce((x, y) => x + y)];
        this.shape = [1];
        return this;
    }

    flatten() {
        this.value = this.value.flat(this.shape.length);
        return this;
    }

    copy() {
        return new Tensor(Array.from(this.value));
    }

    toString() {
        return "Tensor: " + JSON.stringify(this.value) + " requires_grad = " + this.require_grad;
    }
}

let x = new Tensor([[1, 2, 3], [4, 5, 6]]);
let w = new Tensor([1, 2, 3, 4, 5, 6]);

x.multiply(w.reshape([2, 3]));
x.repeat([10]);
x.sumAll()
x.subtract(2).multiply(x);
console.log(x.reshape([1]).toString());
console.log(x.toString());
