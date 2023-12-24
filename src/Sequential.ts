/// <reference path="Model.ts" />

class Sequential extends Model {

    private _layers: Layer[] = [];

    public get layers(): Layer[] {
        return this._layers;
    }

    public add(value: Layer) {
        this._layers.push(value);
    }

    forward(params: Tensor[]): Tensor {
        let res: Tensor;
        this._layers.forEach((layer, i) => {
            if (i == 0) res = layer.forward(params[0]);
            else res = layer.forward(res);           

        })

        return res;
    }

    getParameters(): Map<String, Tensor>[] {
        return this.layers.map((layer) => layer.parameters)
    }



}