function F_init = deepnetinit(Layersizes,Layertypes,Decays)
% Layersizes is a vector containing [Din, H1, H2, ..., Dout].
% Layertypes is a cell array containing unit activations for H1, H2, ...,
%   output.
% Decays is a vector containing the weight decay parameters for each layer.

Nlayers = length(Layersizes)-1;

if ~exist('Decays','var') || isempty(Decays)
  Decays = zeros(1,Nlayers);
end

if length(Layertypes)~=Nlayers
  error('Layertypes has a length not consistent with Layersizes!');
end

if length(Decays)~=Nlayers
  error('Weight decay parameters has a length not consistent with Layersizes!');
end

F_init = cell(1,Nlayers);
for j=1:Nlayers
  layer.type = Layertypes{j};
  layer.l = Decays(j);
  fan_in = Layersizes(j);
  fan_out = Layersizes(j+1);
  layer.units = fan_out;
  layer.W = zeros(fan_in+1,fan_out);
  switch layer.type
    case 'tanh'     % Suggested by Yoshua Bengio, normalized initialization.
      layer.W(1:end-1,:) = 2*(rand(fan_in,fan_out)-0.5)*sqrt(6)/sqrt(fan_in+fan_out);
    case 'cubic'     % Suggested by Yoshua Bengio, normalized initialization.
      layer.W(1:end-1,:) = 2*(rand(fan_in,fan_out)-0.5)*sqrt(6)/sqrt(fan_in+fan_out);
    case 'relu'     % Suggested by Yoshua Bengio, normalized initialization.
      layer.W(1:end-1,:) = 2*(rand(fan_in,fan_out)-0.5)*sqrt(6)/sqrt(fan_in+fan_out);
      % Give some small postive bias so that initial activation is nonzero.
      layer.W(end,:) = rand(1,fan_out)*0.1;
    case 'sigmoid'  % Suggested by Yoshua Bengio, 4 times bigger.
      layer.W(1:end-1,:) = 8*(rand(fan_in,fan_out)-0.5)*sqrt(6)/sqrt(fan_in+fan_out);
    otherwise       % The 1/sqrt(fan_in) rule, small random values.
      layer.W(1:end-1,:) = 2*(rand(fan_in,fan_out)-0.5)/sqrt(fan_in);
  end
  % Bias weights are set to 0.
  F_init{j} = layer;
end
