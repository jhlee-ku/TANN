function nn = nnapplygrads(nn)
% NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases
    
% Update W with dw with L1, L2 penalty term and momentum.
for i = 1 : (nn.n - 1)
        
	% weight sparsity 2015.02.06
   if length(nn.weightPenaltyL2) == 1
     	pl2 = nn.weightPenaltyL2(1);
   else
      pl2 = nn.weightPenaltyL2(i);
   end
        
  	if pl2 > 0
     	dW = nn.dW{i} + pl2 * [zeros(size(nn.W{i}, 1),1) nn.W{i}(:,2:end)];
  	else
      dW = nn.dW{i};
   end
        
		  %keyboard;
	if nn.targethoyersp(i) > 0,
      pl1 = nn.weightPenaltyL1{i};

%   	dW = dW + pl1 * [zeros(size(nn.W{i}, 1),1) sign(nn.W{i}(:,2:end))];
%      dW = (1-pl1)*dW + pl1 * [zeros(size(nn.W{i}, 1),1) sign(nn.W{i}(:,2:end))]; % modified to reflect importance of mse/L1-norm term by adaptively controlling alpha/beta parameters
      %dW = (1-mean(pl1))*dW + diag(pl1) * [zeros(size(nn.W{i}, 1),1) sign(nn.W{i}(:,2:end))]; % reflect node-wise sparsity control
%      dW = diag(1-pl1)*dW + diag(pl1) * [zeros(size(nn.W{i}, 1),1) sign(nn.W{i}(:,2:end))]; % node-wise sparsity control
      dW = (1-pl1)*dW + pl1*[zeros(size(nn.W{i}, 1),1) sign(nn.W{i}(:,2:end))]; % layer-wise sparsity control
   end    
       
   % muliplied with learning rate
  	dW = nn.learningRate(i) * dW;
        
   % momentum
 	if(nn.momentum>0)
     	nn.vW{i} = nn.momentum*nn.vW{i} + dW;
     	dW = nn.vW{i};
  	end
            
  	% gradient descent
  	nn.W{i} = nn.W{i} - dW;
end

%clear dW

end
