if 0
  clear all; clear globals; 
  gaussian = @(r) exp(-1/2 * r.^2);
  gbloaddata;
  P=P';
 [N,dim]=size(P);
end

H=0.10;

rounds =20;


rho = 4;  % bandwidth multiples
for jj=1:rounds

  P0 = P(:,randi(N,1)); 
  R0 = distance(P,P0)/H;
  nidx = find( R0 <= rho);
  fidx = find( R0 > rho);
  fprintf('%d points cannot be truncated\n',length(nidx));
  fprintf('relative near (<rho), relative far(>rho) magnitude %.2e %.2e\n',...
          norm(gaussian(R0(nidx)))/norm(gaussian(R0)), norm(gaussian(R0(fidx)))/norm(gaussian(R0)));
  hidx = find( (R0(:)<=2)&(R0(:)>=0.5) );
  fprintf('Num points in the high derivative region=%d\n',length(hidx));

end
