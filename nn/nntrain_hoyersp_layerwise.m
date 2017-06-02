function [nn, L]  = nntrain(nn, train_x, train_y, val_x, val_y)

% assert(X, A) == if(~X) error(A)
assert(isfloat(train_x), 'train_x must be a float');                            % why input float?
assert(nargin == 3 || nargin == 5,'number of input arguments must be 3 or 5')    % 'nargin' is # of parameters.

loss.train.e               = [];    % training loss function
loss.train.e_frac          = [];    % training error rate
%loss.train.abser=[]; % training absolute error; JH Lee, Nov. 19, 2015
loss.train.varerr=[]; % ratio of variance between input and output signal; for signal prediction; JH Lee, Jan. 03, 2017
loss.val.e                 = [];    % validation loss function
loss.val.e_frac            = [];    % validation error rate
%loss.val.abser=[]; % validation absolute error; JH Lee, Nov. 19, 2015
loss.val.varerr=[]; % JH Lee, Jan. 03, 2017
nn.validation = 0;

if nargin == 5
    nn.validation = 1;
end

% plot figure handle
fhandle = [];
if isfield(nn,'plot') && nn.plot == 1
    fhandle = figure();
end

m = size(train_x, 1);   % m = # of training samples

batchsize = nn.batchsize; % batch size
numepochs = nn.numepochs; % # of epochs
%numbatches = m / batchsize; % # of batches
numbatches = floor(m / batchsize); % # of batches

betarate=nn.betarate; 
decayrate=nn.decayrate;
mlrate=nn.mlrate;

%assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

L = zeros(numepochs*numbatches,1);  % L = sum of squared error "in minibatches". For stochastic gradient descent?
n = 1;                              % for all batches for all samples

for i = 1 : numepochs
    
   tic;
   kk = randperm(m);   	% kk : random training index
   for l = 1 : numbatches
		
   	batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);  % train_x to batch_x
        
      %Add noise to input (for use in denoising autoencoder) ?
      if(nn.inputZeroMaskedFraction ~= 0)
      	batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
     	end
        
     	batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);	% train_y to batch_y

		%keyboard;
		%==============================================================
		% exclude the 'nan' rows; % Apr. 05, 2017 by JHL
		ids_nnan = find(~isnan(batch_x(:,1)) .* ~isnan(batch_y(:,1)));
		batch_x = batch_x(ids_nnan,:); batch_y = batch_y(ids_nnan,:);
		%==============================================================
        
     	nn = nnff(nn, batch_x, batch_y);   % Feed-forward
     	nn = nnbp(nn);                     % Back-propagation
     	nn = nnapplygrads_hoyersp_layerwise(nn);         % SGD; adaptive alpha/beta for node-wise weight sparsity control
     
		%keyboard;
     	for j = 1 : (nn.n-1)                %% nn.n : # of layers 
        	if nn.targethoyersp(j) > 0 
%				if length(nn.weightPenaltyL1{j}) == 1
%					pl1 = nn.weightPenaltyL1{j}(1); % same L1-norm parameter across nodes
%      		else
      			pl1 = nn.weightPenaltyL1{j}; % different L1-norm parameter across nodes
%   			end                        
		
     			Wmtx=nn.W{j}(:,2:end); 
				hsp_Wmtx{j}=hoyersp(Wmtx(:)); % Hoyer's sparseness node-wise

            pl1 = pl1 + betarate*sign(nn.targethoyersp(j) - hsp_Wmtx{j}); % accelerated beta control from Hoyer's sparseness
            %
            pl1=max(0,min(nn.mxbeta(j),pl1));
            nn.weightPenaltyL1{j} = pl1;

				% save mean non-zero ratio along epochs; JH Lee, Mar. 20, 2015
				nn.mhoyersp{j}=[nn.mhoyersp{j}; hsp_Wmtx{j}]; 
				% save weight sparstiy parameter (i.e., betarate) for each epoch; JH Lee, Mar.19,2015
				nn.pl1{j}=[nn.pl1{j}; nn.weightPenaltyL1{j}];
         end    
  		end
    
  		L(n) = nn.l;        % loss function with all batches      
%  		L(n) = gather(nn.l);        % loss function with all batches      

		Varerr(n) = nn.varerr; % error via variance ratio; JH Lee, Jan. 03, 2017
   	n = n + 1;          % all batches for all samples
	end    
  	t = toc;

	str_perf=[];
  	if nn.validation == 1
     	loss = nneval(nn, loss, train_x, train_y, val_x, val_y);
%     	str_perf = sprintf('; train e = %f, val e = %f', loss.train.e(end), loss.val.e(end));
     	str_perf = sprintf('; tr varerr = %f, val e = %f', loss.train.varerr(end), loss.val.varerr(end));
  	else
%      loss = nneval(nn, loss, train_x, train_y); % commented to prevent memory overflow issue
%%      str_perf = sprintf('; train e = %f', loss.train.e(end));
%      str_perf = sprintf('; tr varerr = %f', loss.train.varerr(end));
  	end
  	if ishandle(fhandle)
     	nnupdatefigures(nn, fhandle, loss, nn, i);
  	end
    
   % nn.loss is for different purpose, but is used for saving learning curve
	nn.L = [nn.L mean(L((n-numbatches):(n-1)))];

	nn.Varerr = [nn.Varerr mean(Varerr((n-numbatches):(n-1)))];

	%keyboard;
   % check learning status
	if(rem(i,1)==0),
		mtargethoyersp_str=[]; pl1_str=[]; lrate_str=[];
	   for j = 1 : (nn.n-1)                %% nn.n : # of layers 
        	if nn.targethoyersp(j) ~= 0 
				mtargethoyersp_str=[mtargethoyersp_str, sprintf('%.2f ',mean(hsp_Wmtx{j}))];
				pl1_str=[pl1_str, sprintf('%.3f ',mean(nn.weightPenaltyL1{j}))];
				lrate_str=[lrate_str, sprintf('%.1e ',nn.learningRate(j))];
			end
		end
		
%	   disp(['epoch ' num2str(i) '/' num2str(nn.numepochs) '. Took ' num2str(t) ' seconds' '. Mse of min-batch is ' num2str(mean(L((n-numbatches):(n-1)))) str_perf '; ' sprintf('; lrate=[%s]; sp=[%s]; lrateL1=[%s]',lrate_str,mtargethoyersp_str,pl1_str) ]);
	   disp(['epoch ' num2str(i) '/' num2str(nn.numepochs) '. Took ' num2str(t) ' s' '. mini-batch varerr is ' num2str(nn.Varerr(end)) str_perf '; ' sprintf('; lrate=[%s]; sp=[%s]; lrateL1=[%s]',lrate_str,mtargethoyersp_str,pl1_str) ]);
	end
	    
   % For convergence with adaptvie learning rate
	if (nn.beginAnneal > 0) & (i > nn.beginAnneal),
    	nn.learningRate = max( mlrate, ( decayrate*i+(1-decayrate*nn.beginAnneal) ) * nn.learningRate ); 
	else,
%     	nn.learningRate = nn.learningRate * nn.scaling_learningRate;
     	nn.learningRate = max( mlrate, nn.learningRate * nn.scaling_learningRate );
   end
	nn.lrate = [nn.lrate; nn.learningRate];
end
nn.loss = loss;
%
end
