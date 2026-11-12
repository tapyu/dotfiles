classdef PreluLayer < nnet.layer.Layer ...
    & nnet.internal.cnn.layer.Traceable & nnet.internal.cnn.layer.CPUFusableLayer
    
    %   Copyright 2018-2023 The MathWorks, Inc.
    properties (Learnable)
        Alpha           % Alpha can be a scalar, or [1 ... 1 C]
    end
    
    properties(Hidden, SetAccess=protected)
        NumChannels
        ChannelDim      
    end
    
    methods
        function this = PreluLayer(name, numChannels, channelDim, initialAlpha)
            % layer = PreluLayer(name, numChannels, channelDim,
            % initialAlpha). numChannels can be 1, in which case a single
            % alpha value is shared for all channels, or it can be greater
            % than 1 in which case it must match the number of input
            % channels to the layer. When numChannels>1, initialAlpha can
            % be a scalar or a vector. channelDim is the index of the
            % channel dimension in the input tensor.
            assert(isstring(name) || ischar(name), ...
                message('nnet_cnn_onnx:onnx:LayerNameArg'));
            assert(~isempty(numChannels) && isnumeric(numChannels) && isreal(numChannels) && isscalar(numChannels) && numChannels>0, ...
                message('nnet_cnn_onnx:onnx:PreluArg2'));
            assert(~isempty(channelDim) && isnumeric(channelDim) && isreal(channelDim) && isscalar(channelDim) && channelDim>0, ...
                message('nnet_cnn_onnx:onnx:PreluArg3'));
            assert(~isempty(initialAlpha) && isnumeric(initialAlpha) && isreal(initialAlpha) && ismember(numel(initialAlpha), [1 numChannels]), ...
                message('nnet_cnn_onnx:onnx:PreluArg4'));
            this.Name           = name;
            this.NumChannels    = numChannels;
            this.ChannelDim     = channelDim;
            this.Description    = getString(message('nnet_cnn_onnx:onnx:PreluLayerDescription'));
            this.Type           = getString(message('nnet_cnn_onnx:onnx:PreluLayerType'));
            if this.NumChannels==1
                % Scalar alpha is used for all channels
                this.Alpha      = single(initialAlpha);                
            else
                shape           = [ones(1,this.ChannelDim-1) this.NumChannels 1];    % Need the trailing 1 in case ChannelDim=1
                this.Alpha     	= zeros(shape, 'single');
                this.Alpha(:)   = initialAlpha;
            end
        end
        
        function Z = predict(layer, X)
            
            %Typecasting to workaround issue caused by dlnetwork datatype
            Z = max(0,X) + cast(layer.Alpha,'like',X).*min(0,X);
        end
        
        function [dLdX, dLdAlpha] = backward(this, X, ~, dLdZ, ~)
            chDimIdx    = this.ChannelDim;
            if this.NumChannels == 1
              chDimIdx = 0;  % This will result in a sum over all dims.
            end
            dLdX        = cast(this.Alpha,'like',dLdZ) .* dLdZ;
            posIdx      = X>0;
            dLdX(posIdx)= dLdZ(posIdx);
            nDims       = max(ndims(dLdZ), chDimIdx);
            dimsToSum   = [1:chDimIdx-1, chDimIdx+1:nDims];
            dLdAlpha    = sum(min(0,X).*dLdZ, dimsToSum);
        end
    end

    methods (Static = true, Access = public, Hidden = true)       
         function name = matlabCodegenRedirect(~)
             name = 'nnet.internal.cnn.coder.onnx.PreluLayer';
         end
    end 

    methods(Hidden)
        function layerArgs = getFusedArguments(this)
            %getFusedArguments  Return arguments needed to call the
            % layer in a fused network
            layerArgs = { 'prelu', this.Alpha };
        end

        function tf = isFusable(~)
            %isFusable Flag if layer is fusable
            tf = true;
        end
    end
end
