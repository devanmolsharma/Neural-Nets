type NestedArray<T> = T | NumArray[];
interface NumArray extends Array<NestedArray<number>> { }


class Tensor {
    public gradientHandler: GradientHandler = new GradientHandler(this);

    constructor(private _value: NumArray) {
    }


    public get value(): NumArray {
        return structuredClone(this._value);
    }


    public set value(value: NumArray) {
        this._value = value;
    }

    public get shape(): number[] {
        return this._calculateShape(this._value);
    }

    public clone(){
        return new Tensor(this.value)
    }

    private _calculateShape(array: any, shape: number[] = []): number[] {
        shape.push(array.length);
        if (isNaN(array[0])) {
            this._calculateShape(array[0], shape);
        }
        return shape as number[];
    }

    public backward(gradient: Tensor) {
        this.gradientHandler.backward(gradient)
    }

    public toString():string{
return "Tensor: "+JSON.stringify(this.value)+" shape: "+this.shape
    }

}