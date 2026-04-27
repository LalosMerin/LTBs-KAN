class LTBs_KAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=8, spline_order=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i, (in_f, out_f) in enumerate(zip(layers_hidden, layers_hidden[1:])):
            self.layers.append(KANLinearNS_FactorizedLinear(in_f, out_f, grid_size=grid_size, spline_order=spline_order))
            if i < len(layers_hidden) - 2:
                self.layers.append(nn.Dropout(dropout))  # sin cambio de complejidad asintótica

    def forward(self, x, update_grid: bool = False):
        for layer in self.layers:
            if isinstance(layer, KANLinearNS_FactorizedLinear):
                if update_grid:
                    layer.update_grid(x)
                x = layer(x)
            else:
                x = layer(x)
        return x


        ## Training


        