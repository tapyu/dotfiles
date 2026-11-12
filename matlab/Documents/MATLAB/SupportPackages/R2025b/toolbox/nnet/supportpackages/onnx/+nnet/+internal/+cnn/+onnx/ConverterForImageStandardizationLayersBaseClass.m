classdef ConverterForImageStandardizationLayersBaseClass < nnet.internal.cnn.onnx.NNTLayerConverter
    % A DLT layer that does Y = Scale.*X + Offset where X is [H W C N] or [H W D C N], and
    % Scale and Offset are Cx1. If Offset~=0, it is exported as Mul(Scale)
    % followed by Add(Offset). If Offset==0. it is just Mul(Scale).
    
    % Copyright 2018-2021 The MathWorks, Inc.
    
    properties
        Scale
        Offset
    end
    
    methods
        function this = ConverterForImageStandardizationLayersBaseClass(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function this = set.Scale(this, Scale)
            assert(isnumeric(Scale) && iscolumn(Scale),...
                message('nnet_cnn_onnx:onnx:ImageStandardizationScaleOrOffset'));
            this.Scale = Scale;
        end
        
        function this = set.Offset(this, Offset)
            assert(isnumeric(Offset) && iscolumn(Offset),...
                message('nnet_cnn_onnx:onnx:ImageStandardizationScaleOrOffset'));
            this.Offset = Offset;
        end
        
        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = toOnnx(this, nodeProto, TensorNameMap, TensorLayoutMap)
            import nnet.internal.cnn.onnx.*
            % Implementation of Y=Scale.*X+Offset: Mul(Scale) followed by Add(Offset).
            
            C = numel(this.Scale);
            
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            inputTensorNames    = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});

            if strcmp(inputTensorLayout, 'nchw')
                dims = [1 C 1 1];
            elseif strcmp(inputTensorLayout, 'nchwd')
                dims = [1 C 1 1 1];
            else
                error(message('nnet_cnn_onnx:onnx:UnexpectedONNXInputLayout', this.NNTLayer.Name));
            end
            
            existingNodeNames      = {nodeProto.name};
            % (1) Mul
            MulNodeName            = [onnxName, '_Mul'];
            MulNodeName            = makeUniqueName(existingNodeNames, MulNodeName);            
            MulBName               = [MulNodeName '_B'];
            newNodes(1)           = NodeProto;
            newNodes(1).op_type   = 'Mul';
            newNodes(1).name      = MulNodeName;
            newNodes(1).input     = mapTensorNames(this, {this.InputLayerNames{1}, MulBName}, TensorNameMap);
            newNodes(1).output    = {MulNodeName};
            if this.OpsetVersion < 7
                newNodes(1).attribute = makeAttributeProto('broadcast', 'INT', 1);
            end
            % Initializer for Scale
            t1              = TensorProto;
            t1.name         = MulBName;
            t1.data_type    = TensorProto_DataType.FLOAT;
            t1.raw_data     = rawData(single(this.Scale));
            t1.dims         = dimVector(dims, numel(dims));      % ONNX images have shape [N C H W] or [N C H W D]
            
            parameterInitializers(1) = t1;
            outputTensorName         = MulNodeName;
            
            % (2) If Offset~=0, also generate an Add operator: Adds two inputs Scale
            % and Offset. We'll set Scale to be the output of the Mul and set Offset to
            % be the bias. We'll give Offset an initializer.
            if any(this.Offset ~= 0)
                AddNodeName            = [onnxName, '_Add'];
                AddNodeName            = makeUniqueName([existingNodeNames, {MulNodeName}], AddNodeName);  
                AddBName               = [AddNodeName, '_B'];
                newNodes(2)           = NodeProto;
                newNodes(2).op_type   = 'Add';
                newNodes(2).name      = AddNodeName;
                newNodes(2).input     = {MulNodeName, AddBName};
                newNodes(2).output    = {AddNodeName};
                if this.OpsetVersion < 7
                    newNodes(2).attribute = makeAttributeProto('broadcast', 'INT', 1);
                end
                % Initializer for Offset
                t2              = TensorProto;
                t2.name         = AddBName;
                t2.data_type    = TensorProto_DataType.FLOAT;
                t2.raw_data     = rawData(single(this.Offset));
                t2.dims      	= dimVector(dims, numel(dims));      % ONNX images have shape [N C H W] or [N C H W D]
                
                parameterInitializers(2) = t2;
                outputTensorName         = AddNodeName;
            end
            
            nodeProto               = [nodeProto newNodes];
            networkInputs        	= [];
            networkOutputs        	= [];
            
            % Update maps
            TensorNameMap(this.NNTLayer.Name) = outputTensorName;
            TensorLayoutMap(outputTensorName) = inputTensorLayout;
            
            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end
    end
end
