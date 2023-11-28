/**
 * A simplified implementation of a tensor class.
 */

class Tensor {

    /**
     * Constructor for the Tensor class.
     * @param {Array} value The initial value of the tensor.
     * @param {Boolean} require_grad Specifies whether gradient computation is needed.
     * @param {Array} children An array containing child tensors (used for autograd).
     * @param {Array} backFunctions An array containing backpropagation functions (used for autograd).
     */
    constructor(value, require_grad = true, children = [], backFunctions = []) {
        if (!value) throw ("value not defined");
        this.value = value;
        this.shape = value.length > 0 ? Tensor.calculateShape(value) : [0];
        this.require_grad = require_grad;
        this.gradientFunctions = [];
        this.children = children;
        this.backFunctions = backFunctions;
        this.parent_grads = [];
    }

    /**
     * Register children tensors.
     * @param {Array} children An array containing child tensors.
     */
    registerChildren(children) {
        for (const child in children) {
            this.children.push(child);
        }
    }

    /**
     * Calculate the shape of a multi-dimensional array recursively.
     * @param {Array} array The input array.
     * @param {Array} shape The array used to store the shape information.
     * @returns {Array} The calculated shape of the array.
     */
    static calculateShape(array, shape = []) {
        shape.push(array.length);
        if (isNaN(array[0])) {
            Tensor.calculateShape(array[0], shape);
        }
        return shape;
    }

    /**
     * Add a gradient function to the tensor.
     * @param {Function<x>} fun The gradient function for element-wise operations.
     * @param {Function<x>} fun2 The gradient function for element-wise operations on a second tensor.
     * @param {Tensor} tensor2 The second tensor used for gradient computation.
     */
    addGradientFunction(fun, fun2, tensor2) {
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

    /**
     * Create a new tensor filled with ones and the same shape as the given array.
     * @param {Array} shapeArr The shape of the tensor as an array.
     * @param {Boolean} base Specifies whether to return a new Tensor instance (true) or a plain array (false).
     * @returns {Tensor|Array} A new tensor (or array) filled with ones and the specified shape.
     */
    static onesLike(shapeArr, base = true) {
        if (!base && shapeArr.length == 1) return new Array(shapeArr[0]).fill(1);
        else if (shapeArr.length == 1) return new Tensor(new Array(shapeArr[0]).fill(1));
        let arr = new Array(shapeArr[0]).fill(Tensor.onesLike(shapeArr.slice(1)), false);
        return base ? new Tensor(arr) : arr;
    }

    resetGrads() {
        this.gradients = null;
        this.parent_grads = [];
        this.gradientFunctions = [];
    }

    /**
     * Perform backward pass for gradient computation.
     */
    backward() {
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

    /**
     * Get the gradients computed during the backward pass.
     * @returns {Tensor} The gradients computed during the backward pass.
     */
    getGradients() {
        let gradients;
        if (!gradients && this.parent_grads != []) {
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

    /**
     * Clear the gradients for the tensor.
     */
    clearGradients() {
        this.gradientFunctions = [];
    }

    /**
     * Perform a given operation on the tensor.
     * @param {Function} func The operation to perform.
     * @param {Tensor} addArr Additional array required for the operation.
     * @param {Array} rem_shape Used for further recursion.
     * @param {Array} rem_arr Used for further recursion.
     * @returns {Array} Result of the operation.
     */
    doOperation(func, addArr = null, rem_shape = null, rem_arr = null) {
        let newArr = [];

        // On the first call, set variables to desired values.
        if (!rem_shape) {
            rem_shape = this.shape;
            rem_arr = this.value;
        }

        // While the length of the remaining shape is not 1, it is 2D, so keep passing until a 1D shape is reached.
        if (rem_shape.length > 1) {
            for (let i = 0; i < rem_shape[0]; i++) {
                addArr ? newArr.push(this.doOperation(func, addArr[i], rem_shape.slice(1), rem_arr[i])) :
                    newArr.push(this.doOperation(func, null, rem_shape.slice(1), rem_arr[i]));
            }
            return newArr;
        }

        // When the array is 1D, do the desired calculations and give output.
        for (let index = 0; index < rem_arr.length; index++) {
            newArr.push(func(rem_arr[index], addArr ? addArr[index] : null));
        }
        return newArr;
    }

    /**
     * Perform element-wise operations between this tensor and another tensor (or number).
     * @param {Tensor|Number} tensor2 The second tensor (or number) to perform the operation with.
     * @param {Function} tensorFun The operation function to apply on the tensors.
     * @param {Boolean} returnSelf Specifies whether to return a new tensor or modify the current one.
     * @param {Function} bf1 The backpropagation function for the current tensor.
     * @param {Function} bf2 The backpropagation function for the second tensor.
     * @returns {Tensor|Array} The result of the operation.
     */

    operation(tensor2, tensorFun, returnSelf = true
        , bf1, bf2) {

        // console.log(tensorFun,tensor2.toString());

        // If tensor2 is a tensor, do operations using tensors.
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
        }

        // If it's a number, use the desired settings.
        else if (!isNaN(tensor2)) {
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
    }

    /**
     * Add another tensor (or number) to this tensor.
     * @param {Tensor|Number} tensor2 The other tensor (or number) to add.
     * @returns {Tensor} The final tensor after addition.
     */
    add(tensor2) {
        this.operation(tensor2, (x, y) => x + y);

        if (!(tensor2 === this)) {
            this.addGradientFunction((x, y) => 1, (x, y) => 1, tensor2);
        } else {
            this.addGradientFunction((x, y) => 2, (x, y) => 1, tensor2);
        }

        return this;

    }

    /**
     * Add two tensors element-wise.
     * @param {Tensor} tensor1 The first tensor to add.
     * @param {Tensor} tensor2 The second tensor to add.
     * @returns {Tensor} A new tensor which is the sum of both tensors.
     */
    static add(tensor1, tensor2) {
        return tensor1.operation(tensor2, (x, y) => x + y, false, e => e, e => e);
    }

    /**
     * Multiply this tensor by another tensor (or number).
     * @param {Tensor|Number} tensor2 The other tensor (or number) to multiply with.
     * @returns {Tensor} The final tensor after multiplication.
     */
    multiply(tensor2) {
        if (!(tensor2 === this)) {

            if (isNaN(tensor2) && tensor2.shape.flat() == 1) {
                tensor2 = tensor2.value.flat()[0];
            }

            if (isNaN(tensor2) && this.shape.flat() == 1) {
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

    /**
     * Multiply two tensors element-wise.
     * @param {Tensor} tensor1 The first tensor to multiply.
     * @param {Tensor} tensor2 The second tensor to multiply.
     * @returns {Tensor} A new tensor which is the element-wise product of both tensors.
     */
    static multiply(tensor1, tensor2) {
        return tensor1.operation(
            tensor2,
            (x, y) => x * y,
            false,
            (e) => e.multiply(tensor2 instanceof Tensor ? tensor2.copy() : tensor2),
            (e) => e.multiply(tensor1.copy())
        );
    }

    /**
     * Divide this tensor by another tensor (or number).
     * @param {Tensor|Number} tensor2 The other tensor (or number) to divide by.
     * @returns {Tensor} The final tensor after division.
     */
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

    /**
     * Divide two tensors element-wise.
     * @param {Tensor} tensor1 The first tensor to divide.
     * @param {Tensor} tensor2 The second tensor to divide by.
     * @returns {Tensor} A new tensor which is the element-wise division of both tensors.
     */
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

    /**
     * Reshape the tensor to a new shape.
     * @param {Array} newShape The new shape for the tensor.
     * @returns {Tensor} A new tensor with the specified shape.
     */
    reshape(newShape) {

        if (this.require_grad) {
            let oldshape = this.shape;
            this.addGradientFunction((grad) => grad.reshape(oldshape));
        }

        this.value = this.reshapeValue(this.value, newShape);
        this.shape = newShape;

        return this;
    }

    /**
     * Helper function to recursively reshape a tensor's value.
     * @param {Array} value The value of the tensor.
     * @param {Array} newShape The new shape.
     * @returns {Array} The reshaped value.
     */
    reshapeValue(value, newShape) {
        if (newShape.length === 1) {
            return value;
        }

        const [dim, ...restShape] = newShape;
        const result = [];
        const step = Math.floor(value.length / dim);

        for (let i = 0; i < dim; i++) {
            result.push(this.reshapeValue(value.slice(i*step, (i+1)*step), restShape));
        }

        return result;
    }

    /**
     * Repeat the tensor along specified axes.
     * @param {Array} repeats The number of times to repeat the tensor along each axis.
     * @returns {Tensor} A new tensor repeated along the specified axes.
     */
    repeat(repeats) {

        if (this.require_grad) {
            this.gradientFunctions.push((grad) => Tensor.gradRepeat(grad, this.shape, repeats));
        }

        this.value = this.repeatValue(this.value, repeats);
        this.shape = Tensor.calculateShape(this.value);

        return this;
    }


    /**
     * Helper function to recursively repeat a tensor's value.
     * @param {Array} value The value of the tensor.
     * @param {Array} repeats The number of times to repeat the tensor along each axis.
     * @returns {Array} The repeated value.
     */
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



    /**
     * Subtract another tensor (or number) from this tensor.
     * @param {Tensor|Number} tensor2 The other tensor (or number) to subtract.
     * @returns {Tensor} The final tensor after subtraction.
     */
    subtract(tensor2) {
        this.addGradientFunction((x, y) => 1, (x, y) => -1, tensor2);
        return this.operation(tensor2, (x, y) => x - y);
    }

    /**
     * Subtract two tensors element-wise.
     * @param {Tensor} tensor1 The first tensor to subtract.
     * @param {Tensor} tensor2 The second tensor to subtract.
     * @returns {Tensor} A new tensor which is the element-wise difference of both tensors.
     */
    static subtract(tensor1, tensor2) {
        return tensor1.operation(tensor2, (x, y) => x - y, false, e => e, e => e.multiply(-1));
    }

    /**
     * Subtract two tensors element-wise.
     * @param {Tensor} tensor1 The first tensor to subtract.
     * @param {Tensor} tensor2 The second tensor to subtract.
     * @returns {Tensor} A new tensor which is the element-wise difference of both tensors.
     */
    static powerE(tensor1) {
        let tensor2 = 2.718281828459045235360;
        return tensor1.operation(tensor2, (x, y) => y ** x, false, e => e ** tensor2, e => 0);
    }

    /**
     * Subtract another tensor (or number) from this tensor.
     * @param {Tensor|Number} tensor2 The other tensor (or number) to subtract.
     * @returns {Tensor} The final tensor after subtraction.
     */
    powerE() {
        let tensor2 = 2.718281828459045235360;
        this.operation(tensor2, (x, y) => y ** x);
        this.addGradientFunction((x, y) => y, (x, y) => 0, this.copy());
        return this;
    }


    /**
     * Subtract two tensors element-wise.
     * @param {Tensor} tensor1 The first tensor to subtract.
     * @param {Tensor} tensor2 The second tensor to subtract.
     * @returns {Tensor} A new tensor which is the element-wise difference of both tensors.
     */
    static sumAll(tensor1) {
        let value = tensor1.value.reduce((x, y) => x + y);
        return new Tensor([value], true, [tensor1], [() => Tensor.onesLike(Array.from(tensor1.shape))]);
    }
    /**
     * Subtract another tensor (or number) from this tensor.
     * @param {Tensor|Number} tensor2 The other tensor (or number) to subtract.
     * @returns {Tensor} The final tensor after subtraction.
     */    sumAll() {
        let shape = Array.from(this.shape);
        this.gradientFunctions.push(() => Tensor.onesLike(shape));
        this.value = [this.value.flat(this.shape.length).reduce((x, y) => x + y)];
        this.shape = [1];
        return this;
    }

    /**
     * Flatten the tensor to a 1D array.
     * @returns {Tensor} The tensor flattened into a 1D array.
     */
    flatten() {
        this.value = this.value.flat(this.shape.length);
        return this;
    }

    /**
     * Create a copy of the tensor.
     * @returns {Tensor} A new tensor that is a copy of the current tensor.
     */
    copy() {
        return new Tensor(Array.from(this.value));
    }

    /**
     * Get the string representation of the tensor.
     * @returns {String} The string representation of the tensor.
     */
    toString() {
        return "Tensor: " + JSON.stringify(this.value) + " requires_grad = " + this.require_grad;
    }
}


let x = new Tensor([[1, 2, 3], [4, 5, 6]]);
let w = new Tensor([1, 2, 3, 4, 5, 6]);

x.multiply(w.reshape([2,3]));
x.repeat([10]);
x.sumAll()
x.subtract(2).multiply(x);
console.log(x.reshape([1]).toString());
console.log(x.toString());