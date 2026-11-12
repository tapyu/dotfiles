classdef ConverterForGRULayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a gruLayer into ONNX

    % Supported input formats: S*CB[T] for the main input and CB for state inputs. 

    % Copyright 2020-2024 The Mathworks, Inc.

    methods
        function this = ConverterForGRULayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] = toOnnx(...
                this, nodeProto, TensorNameMap, TensorLayoutMap)
            % Call into the base method passing the NNTLayer input and
            % recurrent weights.
            [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] = this.convertGRUWithWeightsToONNX(...
                nodeProto, TensorNameMap, TensorLayoutMap, this.NNTLayer.InputWeights, this.NNTLayer.RecurrentWeights);
        end

        function [nodeProto, parameterInitializers, networkInputs, networkOutputs, TensorNameMap, TensorLayoutMap] = convertGRUWithWeightsToONNX(...
                this, nodeProto, TensorNameMap, TensorLayoutMap, passedInputWeights, passedRecurrentWeights)
            % Inherited method used both by ConverterForGRULayer and
            % ConverterForGRUProjectedLayer.

            import nnet.internal.cnn.onnx.*

            existingNodeNames       = {nodeProto.name};
            [gruNodeName, nameChanged]  = legalizeNNTName(this, this.NNTLayer.Name);
            gruNodeName                 = makeUniqueName({nodeProto.name}, gruNodeName);
            NNTLayer                = this.NNTLayer;
            hasSequenceOutput       = isequal(NNTLayer.OutputMode, 'sequence');

            inputTensorNames = mapTensorNames(this, this.InputLayerNames, TensorNameMap);

            inputTensorLayout   = TensorLayoutMap(inputTensorNames{1});

            % Unsqueeze if inputTensorLayout is 'CB'
            if inputTensorLayout == "nc"
                unsqueezeAxis                 = 0;
                unsqueezeName                 = [inputTensorNames{1} '_Unsqueeze'];
                unsqueezeName                 = makeUniqueName(existingNodeNames, unsqueezeName);
                unsqueezeInput                = {inputTensorNames{1}};
                unsqueezeOutput               = {unsqueezeName};
                [unsqueezeNode, unsqueezeInit]  = createNodeProto(this, 'Unsqueeze', unsqueezeName, unsqueezeInput, unsqueezeOutput, unsqueezeAxis);
                gruInputTensorName              = unsqueezeName;
                hasSequenceOutput             = false;
            elseif inputTensorLayout == "snc"
                gruInputTensorName = inputTensorNames{1};
                unsqueezeNode = [];
                unsqueezeInit = [];
            end

            %Set outputTensorLayout
            if hasSequenceOutput
                gruOutputTensorLayout  = 'snc';
            else
                gruOutputTensorLayout  = 'nc';
            end

            % Make the GRU input names
            gruInput               = {...
                gruInputTensorName,... % X
                [gruNodeName '_W'],...
                [gruNodeName '_R'],...
                [gruNodeName '_B'],...
                '',...                      % sequence_Lens
                [gruNodeName '_initial_h']
                };
            if NNTLayer.HasStateInputs
                warning(message('nnet_cnn_onnx:onnx:UnsupportedRNNStateInputs', this.NNTLayer.Name))
                gruInput{6} = inputTensorNames{2};
            end

            % Create nodes to transform the state inputs, if present.
            [StateInputPreprocessingNodes, StateInputPreprocessingInits, gruInput] = createStateInputPreprocessingNodes(...
                this, NNTLayer, gruNodeName, gruInput);

            gruInput = mapTensorNames(this, gruInput(:)', TensorNameMap);

            % Include state outputs if necessary
            if NNTLayer.HasStateOutputs
                gruOutputTensorName     = [gruNodeName '/out'];
                hiddenOutputTensorName  = [gruNodeName '/hidden'];
                if hasSequenceOutput
                    gruOutput      = {gruOutputTensorName hiddenOutputTensorName};
                else
                    gruOutput      = {'', hiddenOutputTensorName};
                end
            elseif hasSequenceOutput
                gruOutputTensorName        = gruNodeName;
                gruOutput          = {gruOutputTensorName};          % Output is Y=GRUonnxNamegru
            else
                gruOutputTensorName        = gruNodeName;
                gruOutput          = {'', gruOutputTensorName};      % Output is Y='', Y_h=onnxName
            end

            % Set attributes
            iofAct	= iTranslateActivationFunction(NNTLayer.GateActivationFunction,this.NNTLayer.Name);
            hidAct	= iTranslateActivationFunction(NNTLayer.StateActivationFunction,this.NNTLayer.Name);

            % Create GRU node
            [gruNode, gruInits] = createNodeProto(this, 'GRUForward', gruNodeName, gruInput, gruOutput, ...
                NNTLayer.InputSize, hasSequenceOutput, ...
                iofAct, hidAct, passedInputWeights, passedRecurrentWeights, NNTLayer.Bias, ...
                NNTLayer.NumHiddenUnits, NNTLayer.ResetGateMode);

            % Follow GRU with Squeeze operators to reshape the outputs
            % Squeeze main output (Y or Y_h):
            if hasSequenceOutput
                squeezeAxis = 1;
            else
                squeezeAxis = 0;
            end
            squeezeName                 = [gruNodeName '_Squeeze'];
            squeezeInput                = {gruOutputTensorName};
            squeezeOutput               = {squeezeName};
            [squeezeNode, squeezeInit]  = createNodeProto(this, 'Squeeze', squeezeName, squeezeInput, squeezeOutput, squeezeAxis);
            gruOutputTensorName         = squeezeName;

            % Maybe Squeeze state outputs from 1NC to NC
            squeezeHiddenNode = [];
            squeezeHiddenInit = [];
            if NNTLayer.HasStateOutputs
                % Squeeze the Y_h output if it isn't the main output:
                if hasSequenceOutput
                    squeezeHiddenName         = [gruNodeName '_SqueezeHidden'];
                    squeezeHiddenName         = makeUniqueName([existingNodeNames, {gruNodeName}], squeezeHiddenName);
                    squeezeHiddenInput        = {hiddenOutputTensorName};
                    squeezeHiddenOutput       = {squeezeHiddenName};
                    [squeezeHiddenNode, squeezeHiddenInit]  = createNodeProto(this, 'Squeeze', squeezeHiddenName, squeezeHiddenInput, squeezeHiddenOutput, 0);
                    hiddenOutputTensorName    = squeezeHiddenName;
                end
            end

            % Collect the appropriate set of nodes and initializers
            nodeProto = [nodeProto unsqueezeNode StateInputPreprocessingNodes gruNode squeezeNode squeezeHiddenNode];
            parameterInitializers = [unsqueezeInit StateInputPreprocessingInits gruInits squeezeInit squeezeHiddenInit];

            networkInputs           = [];
            networkOutputs          = [];

            % Determine layer output port names. These are the names used
            % by the MATLAB representation of the layer
            onnxNamePort = this.NNTLayer.Name;
            gruOutputPort = this.NNTLayer.Name;
            if NNTLayer.HasStateOutputs
                gruOutputPort = [this.NNTLayer.Name '/out'];
                hiddenOutputPort = [this.NNTLayer.Name '/hidden'];
            end

            % Update maps
            TensorNameMap(onnxNamePort) = gruOutputTensorName;
            TensorNameMap(gruOutputPort) = gruOutputTensorName;
            TensorLayoutMap(gruOutputTensorName) = gruOutputTensorLayout;
            if NNTLayer.HasStateOutputs
                TensorNameMap(hiddenOutputPort) = hiddenOutputTensorName;
                TensorLayoutMap(hiddenOutputTensorName) = 'nc';
                if ~hasSequenceOutput
                    TensorNameMap(onnxNamePort) = hiddenOutputTensorName;
                    TensorNameMap(gruOutputPort) = hiddenOutputTensorName;
                    TensorLayoutMap(gruOutputTensorName) = 'nc';
                end
            end

            % Update network if current layer is danglingLayer
            [nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap] ...
                = this.updateNetworkOutputForDanglingLayers(nodeProto, networkOutputs, TensorNameMap, TensorLayoutMap);
        end


        function [nodes, inits, gruInputNames] = createStateInputPreprocessingNodes(this, NNTLayer, gruNodeName, gruInputNames)
            % Create nodes to transform the state inputs prior to GRU
            % node. Update gruInputNames with the names output by these
            % nodes.
            import nnet.internal.cnn.onnx.*
            if NNTLayer.HasStateInputs
                % There are state inputs. Insert unsqueeze nodes.

                % Unsqueeze initial_h from BC to 1BC
                hUnsqNodeName           = [gruNodeName '_Unsqueeze_Hidden_Input'];
                [hUnsqNode, hUnsqInit]  = createNodeProto(this, 'Unsqueeze', hUnsqNodeName, gruInputNames(6), {hUnsqNodeName}, 0);

                nodes               = [hUnsqNode];
                inits               = [hUnsqInit];
                gruInputNames{6}   = hUnsqNodeName;
            else
                % There are no state inputs. Insert nodes that gather the
                % batchSize from X and expand the state initializers to
                % their required shape using Tile. We want to create 1BH in
                % onnx, given Hx1 matlab data and the ONNX X input of shape
                % TBC. The following ONNX code has been carefully chosen so
                % it works for all opsets>=6:
                %
                % S = Shape(X)					                            S is rank 1, int64  = [T B C].
                % B = Gather(S, init([1]), axis=0)			                Make indices [1] (rank 1) to get rank 1 output.
                % repeats = Concat(init([1]), B, init([1]), axis=0).	    Build the vector [1,B,1]
                % newH = Tile(H, repeats)                                   Turn shape [1,1,H] into [1,B,H]
                % newC = Tile(C, repeats)                                   Turn shape [1,1,H] into [1,B,H]

                % Shape(X) node:
                sNodeName = [gruNodeName '_Shape1'];
                [sNode, ~]  = createNodeProto(this, 'Shape', sNodeName, gruInputNames(1), {sNodeName});

                % Gather node:
                gNodeName = [gruNodeName '_Gather1'];
                gIndicesName = [gruNodeName '_Gather1Indices'];
                gInit = nnet.internal.cnn.onnx.makeTensorProtoOfType(gIndicesName, [1], 1, TensorProto_DataType.INT64);
                [gNode, ~]  = createNodeProto(this, 'Gather', gNodeName, {sNodeName, gIndicesName}, {gNodeName}, 0);

                % Concat node:
                cNodeName = [gruNodeName '_Concat1'];
                cConst1Name = [gruNodeName '_ConcatConst1'];
                cConst1Init = nnet.internal.cnn.onnx.makeTensorProtoOfType(cConst1Name, [1], 1, TensorProto_DataType.INT64);
                [cNode, ~]  = createNodeProto(this, 'Concat', cNodeName, {cConst1Name, gNodeName, cConst1Name}, {cNodeName}, 0);

                % Tile(H) node:
                tNodeName1 = [gruNodeName '_TileH'];
                [tNode1, ~]  = createNodeProto(this, 'Tile', tNodeName1, {gruInputNames{6}, cNodeName}, {tNodeName1});

                % Create initializer for the raw initial_h input tensor, of
                % shape [1,1,nH].
                HInit       = makeTensorProtoOfType(gruInputNames{6}, [1,1,NNTLayer.NumHiddenUnits], NNTLayer.HiddenState, TensorProto_DataType.FLOAT);

                % Return the nodes and inits and update the GRU input names
                nodes               = [sNode, gNode, cNode, tNode1];
                inits               = [gInit, cConst1Init, HInit];
                gruInputNames{6}   = tNodeName1;
            end
        end
    end
end

function s = iTranslateActivationFunction(s,layerName)
switch s
    case 'sigmoid'
        s = 'Sigmoid';
    case 'softsign'
        s = 'Softsign';
    case 'tanh'
        s = 'Tanh';
    case 'relu'
        s = 'Relu';
    case 'hard-sigmoid'
        s = 'HardSigmoid';
    otherwise
        error(message('nnet_cnn_onnx:onnx:InvalidActivationFunction', s, layerName));
end
end