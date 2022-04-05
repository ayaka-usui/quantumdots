
using Arpack, SparseArrays, LinearAlgebra
# using ExpmV

function vNEntropy(matC::Matrix{Float64})

    output = 0.0
    jjmax = size(matC,1)

    for jj = 1:jjmax
        output += matC
    end

    return

end

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

    C0 = zeros(Float64,K*2+1)
    C0[1] = 0.0 # n_d(0)

    for kk = 1:K
        C0[1+kk] = 1.0/(exp((matH[1+kk,1+kk]-muL)*betaL)+1.0)
        C0[1+K+kk] = 1.0/(exp((matH[1+K+kk,1+K+kk]-muR)*betaR)+1.0)
    end
    C0 = diagm(C0)
    # C0 = spdiagm(C0)
    # vecC0 = zeros(Float64,K*2+1)
    Ct = zeros(ComplexF64,K*2+1,K*2+1)
    Ct_L = zeros(ComplexF64,K*2+1,K*2+1)
    Ct_R = zeros(ComplexF64,K*2+1,K*2+1)
    val_Ct_L = zeros(ComplexF64,K*2+1)
    val_Ct_R = zeros(ComplexF64,K*2+1)
    vNE_sys = zeros(ComplexF64,Nt)
    vNE_L = zeros(ComplexF64,Nt)
    vNE_R = zeros(ComplexF64,Nt)

    # Hamiltonian is hermitian
    matH = Hermitian(Array(matH))
    val_matH, vec_matH = eigen(matH)
    invvec_matH = inv(vec_matH)

    # correlation matrix
    for tt = 1:Nt

        Ct .= vec_matH*diagm(exp.(1im*val_matH*time[tt]))*invvec_matH
        Ct .= Ct*C0
        Ct .= Ct*vec_matH*diagm(exp.(-1im*val_matH*time[tt]))*invvec_matH
        # Ct = Hermitian(Ct)

        # vNE
        vNE_sys[tt] = -(Ct[1,1]*log(Ct[1,1]) + (1-Ct[1,1])*log(1-Ct[1,1]))

        Ct_L .= Ct[2:K*2+1,2:K*2+1]
        val_Ct_L .= eigvals(Ct_L)
        vNE_L[tt] = - sum(val_Ct_L.*log.(val_Ct_L)) - sum((1.0 .- val_Ct_L).*log.(1.0 .- val_Ct_L))

        Ct_R .= Ct[K*2+2:end,K*2+2:end]
        val_Ct_R .= eigvals(Ct_R)
        vNE_R[tt] = - sum(val_Ct_R.*log.(val_Ct_R)) - sum((1.0 .- val_Ct_R).*log.(1.0 .- val_Ct_R))

    end



end
