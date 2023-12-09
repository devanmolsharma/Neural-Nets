
/// <reference path="TensorOperation.ts" />
// Class representing a layer in neural network
abstract class Layer {
    constructor() {
    }

    // Parameters of the Layer
    private _parameters: Map<String, Tensor> = new Map<String, Tensor>();

    public get parameters(): Map<String, Tensor> {
        return this._parameters;
    }

    public set parameters(value: Map<String, Tensor>) {
        this._parameters = value;
    }

    public registerParameter(name: string, parameter: Tensor) {
        this._parameters.set(name, parameter);
    }

    public abstract forward(tensor: Tensor): Tensor;

    isPrimitive(val: Object): boolean {

        if (val === Object(val)) {
            return false;
        } else {
            return true;
        }
    }


    // Planning on using this to save and load models
    public toJson() {
        let j: Array<any> = [];
        (Object.getOwnPropertyNames(this)).forEach((name: string) => {
            const value: any = (this as any)[name];
            let tempValue: any;
            if (value instanceof Tensor) {
                tempValue = value.value;
            } else if (!this.isPrimitive(value)) {
                return;
            } else {
                tempValue = value;
            }
            j.push({
                'name': name,
                'value': tempValue
            })
        });
        return JSON.stringify(j);
    }

    public loadData(json: string) {
        let params = JSON.parse(json);
        params.forEach((data: any) => {
            if (data.value instanceof Array) {
                (this as any)[data.name] = new Tensor(data.value);
                (this._parameters as any)[data.name] = (this as any)[data.name];

            } else {
                (this as any)[data.name] = data.value;
                (this._parameters as any)[data.name] = (this as any)[data.name];
            }
        });
        return this;
    }
}