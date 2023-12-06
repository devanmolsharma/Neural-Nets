// Class representing a layer in neural network
abstract class Layer {
    // Parameters of the Layer
    private _parameters: Map<String, Tensor>;

    public get parameters(): Map<String, Tensor> {
        return this._parameters;
    }

    public set parameters(value: Map<String, Tensor>) {
        this._parameters = value;
    }

    private registerParameter(name: string, parameter: Tensor) {
        this.parameters.set(name, parameter);
    }

    public abstract forward(tensor:Tensor):Tensor;

    // Planning on using this to save and load models
    public toJson() {
        return JSON.stringify({
            name: this.constructor.name,
            parameters: this.parameters.forEach((tensor, key, map) => { key: tensor.value })
        })
    }
}