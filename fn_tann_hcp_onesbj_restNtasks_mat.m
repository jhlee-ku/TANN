%
% function to run the temporal autoencoding neural network (TANN)
%
function	fn_tann_hcp_onesbj_restNtasks_mat(sbjnm, list_hcp_flnm)
%
%clear all;

rootpth='/data/leej45';
datapth=[rootpth '/s3/hcp']; 
%=== addpath for commonly used m-files
%addpath([rootpth '/_codes']);
%=== addpath for cifti file interface
%addpath([rootpth '/_codes/cifti']);
%addpath([rootpth '/_codes/cifti/gifti-1.6']);
%=== addpath for NN functions
%addpath([rootpth '/_codes/dnn/nn']);
%addpath([rootpth '/_codes/dnn/util']);

%=== load HCP data
%sbjstr='100307'; epistr='rfMRI_REST1_LR'; ciftiflnm=[epistr '_Atlas_hp2000_clean']; nTR=1200;
%sbjstr='100307'; epistr='tfMRI_EMOTION_LR'; ciftiflnm=[epistr '_Atlas']; nTR=176;
%sbjstr='100307'; epistr='tfMRI_MOTOR_LR'; ciftiflnm=[epistr '_Atlas']; nTR=284;
%%
%ciftifl=fullfile('/s3/hcp',sbjstr,'MNINonLinear/Results',epistr,[ciftiflnm '.dtseries.nii']);disp(ciftifl);
%cii=ciftiopen(ciftifl, 'wb_command');

%===========================================================================
%     Fine-tunning of SAE-pretrained DNN for input pattern reconstruction
%===========================================================================
ngrayord=91282; 
ncortsurf=59412; % cortical vertice excluding median wall
nsubcortcerebel=ngrayord-ncortsurf; % subcortical and cerebellum voxels
targethoyersps={[0.7 0.7]};
nhiddenodes=100; % number of hidden nodes
inputnormalize = 'unitvar_per_volumn'; % 'unitvar_per_volumn' or 'unitvar_per_boldts'
X=[];
%	list_sbj_hcp_flnm='list_sbj_hcp_tfMRI_MOTOR.txt';
%	list_sbj_hcp_flnm='list_hcp_100sbjs_tfMRI_MOTOR.txt';
%list_sbj_hcp_flnm='tmp_hcp_rfMRI_REST1_10sbjs.txt';
%list_sbj_hcp_flnm='list_hcp_rfMRI_REST1_MSMAll_50sbjs';
%list_sbj_hcp_flnm='list_hcp_100307_restNtasks_MSMAll_Ssmoothed4';
%fp=fopen([list_sbj_hcp_flnm '.txt'],'rt');
fp=fopen([list_hcp_flnm '.txt'],'rt');
while 1,
	lstr=fgetl(fp); %disp(lstr);
	if(~ischar(lstr)), break, end,
%	cii=ciftiopen(['/data/HCP/s3-hcp-openaccess/' lstr], 'wb_command');
%	ciftiflnm=[rootpth '/s3/hcp/' sbjnm '/' lstr]; disp(ciftiflnm);
%	if(exist(ciftiflnm)),
%		cii=ciftiopen(ciftiflnm, 'wb_command');
%	else,
%		continue;
%	end

	matflnm=[datapth '/' sbjnm '/' lstr]; disp(matflnm);
	load([matflnm '.mat'],'cdata');
	cii.cdata=cdata;
	%
	switch inputnormalize,
		case 'unitvar_per_volumn'
%			x0=cii.cdata - mean(cii.cdata,2)*ones(1,size(cii.cdata,2)); x=reshape(zscore(x0(:)),ngrayord,size(x0,2))'; % to preserve the original BOLD intensity as much possible while removing unwanted BOLD intensity bias across whole brain
%			x0=cii.cdata(1:ncortsurf,:) - mean(cii.cdata(1:ncortsurf,:),2)*ones(1,size(cii.cdata,2)); x=reshape(zscore(x0(:)),ncortsurf,size(x0,2))'; % to preserve the original BOLD intensity as much possible while removing unwanted BOLD intensity bias across the cortical gray matter (surface)
			% remove the bias and variance normalization for each of the cortical/subcort-cerebellum separately
			x0_cortsurf=cii.cdata(1:ncortsurf,:) - mean(cii.cdata(1:ncortsurf,:),2)*ones(1,size(cii.cdata,2));
%			x_cortsurf=reshape(zscore(x0_cortsurf(:)),ncortsurf,size(cii.cdata,2));
			x_cortsurf=(x0_cortsurf - mean(x0_cortsurf(:)))/std(x0_cortsurf(:));
			%
			x0_subcortcerebel=cii.cdata(ncortsurf+1:ngrayord,:) - mean(cii.cdata(ncortsurf+1:ngrayord,:),2)*ones(1,size(cii.cdata,2));
%			x_subcortcerebel=reshape(zscore(x0_subcortcerebel(:)),nsubcortcerebel,size(cii.cdata,2));
			x_subcortcerebel=(x0_subcortcerebel - mean(x0_subcortcerebel(:)))/std(x0_subcortcerebel(:));
			%
			x=[x_cortsurf; x_subcortcerebel]';
			x = x / 10;
			%
	%		x=reshape(zscore(cii.cdata(:)),ngrayord,nTR)';
	%		x=zscore(cii.cdata)'; % normalize intensity of each volume to unit-variance
	%		x=zscore(cii.cdata - mean(cii.cdata,2)*ones(1,size(cii.cdata,2)))'; % deman each volume and normalize to unit-variance
		case 'unitvar_per_boldts'
			x=zscore(cii.cdata'); % normalize intensity of each voxel's BOLD time-series to unit-variance
		otherwise
			error('input data normalization method is wrong.');
	end
	fprintf('size of x = [%d %d]\n',size(x,1),size(x,2));

	X=[X; x]; % concatenating the data
end
fclose(fp);
fprintf('size of X = [%d %d]\n', size(X,1), size(X,2));
%
%keyboard;
%c=parcluster('local');
%c.NumWorkers=3;
%parpool(c, c.NumWorkers);
%
for hoyerspdx=1:length(targethoyersps),
%parfor hoyerspdx=1:length(targethoyersps),
	targethoyersp=targethoyersps{hoyerspdx};
	%
	nn = nnsetup_hoyersp([ngrayord nhiddenodes ngrayord]);
	nn.inputnormalize = inputnormalize; % 'unitvar_per_volumn' or 'unitvar_per_boldts'
	%
	%==================== overall parameter setting
	nn.activation_function = 'tanh_opt'; nn.output = 'linear'; 
	nn.numepochs = 300; nn.beginAnneal = 100; 
	nn.learningRate = [5e-3 5e-3]; nn.scaling_learningRate=1; nn.decayrate=-1e-4; nn.mlrate=1e-6;
	nn.batchsize = 200; 
	nn.momentum=0.5;
	nn.validation = 0; nn.plot = 0;
	%==================== for input nodes
	nn.inputZeroMaskedFraction=0.0; 
	%==================== for hidden nodes
	nn.dropoutFraction=0;
	nn.nonSparsityPenalty=[0]; 
	nn.sparsityTarget=0; 
	%==================== for weight parameters
	nn.sparsity_type = 'hoyer'; % 'nzr' or 'hoyer'
	nn.sparsity_unit= 'layerwise'; % 'layerwise' or 'nodewise'
%	nn.targethoyersp=[0.95 0.95]; % 0 being the lowest and 1 being the highest sparseness
	nn.targethoyersp=targethoyersp;
	nn.weightPenaltyL1{1}=1e-3; nn.weightPenaltyL1{2}=1e-3; % typically, 1e-3
	nn.mxbeta=[0.01 0.01]; nn.betarate=1e-3;
	nn.weightPenaltyL2=[1e-5 1e-5];
	%===========================================================================

	nn,
	disp('now learning..');
	%===========================================================================
	%		Begin training the NN
	%===========================================================================
	switch nn.sparsity_type
		case 'nzr'
			switch nn.sparsity_unit
				case 'nodewise'
					nn = nntrain_nzr_nodewise(nn, X(1:nTR-1,:), X(2:nTR,:));
				case 'layerwise'
					nn = nntrain_nzr_layerwise(nn, X(1:nTR-1,:), X(2:nTR,:));
				otherwise, error('Unit of weight sparsity control is not correctly defined.');
			end
		
		case 'hoyer'
			switch nn.sparsity_unit
				case 'nodewise'
					nn = nntrain_hoyersp_nodewise(nn, X(1:end-1,:), X(2:end,:));
				case 'layerwise'
					nn = nntrain_hoyersp_layerwise(nn, X(1:end-1,:), X(2:end,:));
				otherwise, error('Unit of weight sparsity control is not correctly defined.');
			end

		otherwise
			error('Type of sparsity is not correctly defined');
	end
	%
	%save('rst_load_hcp_cifti.mat','nn');
	sznet_str=sprintf('%d_',nn.size);
	sp_str=sprintf('%1.2f_',nn.targethoyersp);
	%======================================= save the TANN results
	C=[]; C.nn=nn; 
	savflnm=['rst_tann_hcp_onesbj_restNtasks_mat_' sznet_str nn.sparsity_type 'sp' sp_str nn.sparsity_unit '_inputzero' num2str(nn.inputZeroMaskedFraction) ...
					'_dropout' num2str(nn.dropoutFraction) '_hactfn_' nn.activation_function '_oactfn_' nn.output ...
					'_' num2str(nn.numepochs) 'epoch_' num2str(nn.beginAnneal) 'beginAnneal_' ...
					nn.inputnormalize	'_' [sbjnm '_' list_hcp_flnm] '_' timestring];
	parfor_save(fullfile(datapth,sbjnm,[savflnm '.mat']), C);
	%======================================== save encoding/decoding features
	We=nn.W{1}(:,2:end); Wd=nn.W{2}(:,2:end);
%	newcii=cii; newcii.cdata=We'; ciftisave(newcii,[savflnm '_We.dtseries.nii'],'wb_command');
%	newcii=cii; newcii.cdata=Wd; ciftisave(newcii,[savflnm '_Wd.dtseries.nii'],'wb_command');
	%
	signskewness_Wd=sign(skewness(Wd)); % for sign correction of Wd
	Wd_signc=Wd.*(ones(size(Wd,1),1)*signskewness_Wd); % skewness/sign corrected Wd
	%
	%save('rst.mat'),'We','Wd');
	save(fullfile(datapth,sbjnm,[savflnm '_We_Wd_signc.mat']),'We','Wd','Wd_signc'); % save encoding/decoding features
	%=================================================================================================
end
%
end
