declare class GradientHandler {
    private _tensor;
    private Operation;
    private _children;
    private _gradient;
    constructor(_tensor: Tensor);
    get children(): Tensor[];
    get gradient(): Tensor;
    set gradient(value: Tensor);
    registerChildren(...children: Tensor[]): void;
    registerOperation(operation: TensorOperation): void;
    backward(previousGradient?: Tensor): void;
}
type NestedArray<T> = T | NumArray[];
interface NumArray extends Array<NestedArray<number>> {
}
declare class Tensor {
    private _value;
    gradientHandler: GradientHandler;
    constructor(_value: NumArray);
    get value(): NumArray;
    set value(value: NumArray);
    get shape(): number[];
    clone(): Tensor;
    private _calculateShape;
    backward(gradient: Tensor): void;
    toString(): string;
}
declare abstract class TensorOperation extends Function {
    private _bound;
    constructor();
    verify(...tensors: Tensor[]): void;
    abstract forward(...tensors: NumArray[]): NumArray;
    abstract backward(gradient: NumArray): NumArray[];
    private _call;
    getGradient(gradient: Tensor): Tensor[];
}
declare class Sum extends TensorOperation {
    private addArrays;
    private sum;
    forward(...tensors: NumArray[]): NumArray;
    backward(gradient: NumArray): NumArray[];
}
declare class Mean extends TensorOperation {
    forward(...tensors: NumArray[]): NumArray;
    backward(gradient: NumArray): NumArray[];
}
declare const t1: Tensor, t2: Tensor, t3: Tensor;
declare let j: any;
