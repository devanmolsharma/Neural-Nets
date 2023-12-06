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

    // Planning on using this to save and load models
    public toJson() {
        let j: any = [];
        this.parameters.forEach((e, k) => j.push({ name: k, value: e.value }))
        return JSON.stringify({
            'name': this.constructor.name, parameters: j,
        })
    }

    static fromJson(json:string) {
        let params = JSON.parse(json);
        
    }
}