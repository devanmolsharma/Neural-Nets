

// Define a type for nested arrays, which can contain either a value of type T or nested NumArray arrays
type NestedArray<T> = T | Array<NestedArray<T>>;

// Define an interface for NumArray, which extends the Array type and allows nesting of NestedArray<number>
interface NumArray extends Array<NestedArray<number>> { }

/// <reference path="GradinetHandler.ts" />
// Class representing a Tensor
class Tensor {
    // Gradient handler for backpropagation
    public gradientHandler: GradientHandler = new GradientHandler(this);

    // Constructor takes a NumArray as the initial value
    constructor(private _value: NumArray) {
    }

    // Getter for the value, returning a deep clone to avoid unintentional modifications
    public get value(): NumArray {
        return structuredClone(this._value);
    }

    // Setter for the value
    public set value(value: NumArray) {
        if (JSON.stringify(TensorUtils.calculateShape(value)) != JSON.stringify(this.shape)) throw `shapes dont match , setting ${JSON.stringify(value)} to tensor ${JSON.stringify(this.value)} , shapes ${(TensorUtils.calculateShape(value))} , ${JSON.stringify(this.shape)} `;
        this._value = value;
    }

    // Getter for the shape of the tensor
    public get shape(): number[] {
        return this._calculateShape(this._value);
    }

    // Method to create a clone of the tensor
    public clone() {
        return new Tensor(this.value)
    }

    // Private method to recursively calculate the shape of the tensor
    private _calculateShape(array: any, shapeArr: number[] = []): number[] {
        shapeArr.push(array.length);
        if (Array.isArray(array[0])) {
            this._calculateShape(array[0], shapeArr);
        }
        return shapeArr as number[];
    }

    // Method for performing backpropagation
    public backward(gradient: Tensor) {
        this.gradientHandler.backward(gradient)
    }

    // Method to convert the tensor to a string representation
    public toString(): string {
        return "Tensor: " + JSON.stringify(this.value) + " shape: " + this.shape;
    }
}
