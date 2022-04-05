
using Arpack, SparseArrays, LinearAlgebra

function createH!(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,matH::SparseMatrixCSC{Float64})

    # matH = sparse(Float64,K*2+1,K*2+1)
    Depsilon = W/(K-1)

    matH[1,1] = 1.0 # epsilon for the system

    for kk = 1:K
        matH[1+kk,1+kk] = (kk-1)*Depsilon - W/2 # epsilon for the bath L
        matH[1+kk,1] = sqrt(GammaL*Depsilon/(2*pi)) # tunnel with the bath L
        matH[1+K+kk,1] = sqrt(GammaR*Depsilon/(2*pi)) # tunnel with the bath R
    end
    matH[K+2:end,K+2:end] = matH[2:K+1,2:K+1] # epsilon for the bath R

    matH .= matH + matH' - spdiagm(diag(matH))

end

function calculatequantities(K::Int64,W::Int64,betaL::Float64,betaR::Float64,GammaL::Float64,GammaR::Float64,muL::Float64,muR::Float64,tf::Float64,Nt::Int64)

    matH = sparse(Float64,K*2+1,K*2+1)
    createH!(K,W,betaL,betaR,GammaL,GammaR,matH)

    # time
    time = LinRange(0.0,tf,Nt)
    Ct =

    # vNE
    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 # n_d(0)

    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end
    C0 = spdiagm(C0)

    for tt = 1:Nt
        Ct[:,:,tt] = C0[:,:]
    end

end
