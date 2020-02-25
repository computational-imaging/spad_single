# Power to number of photons
N(P, λ, h, c) = P*λ/(h*c)
# Area of square region given FoV
A(θ, r) = 4*r^2*tand(θ/2)^2
# Return power given output power, areas of receivers, distances, etc.
R(P, ρ, A_t, A_illum, A_rec, r, η) = P * (ρ*A_t/A_illum) * (A_rec/(π*r^2)) * η

# Constants
h = 6.626e-34   # Planck's constant (m²kg/s)
c = 3e8         # Speed of light (m/s)

# Laser params
λ = 450e-9      # Operating wavelength (m)
# P_L = 0.5e-3      # Laser power (point scanner) (W)
P_L = 60e-3     # Kinect uses a 60mW laser (W)
f = c/λ         # Frequency
N_L = N(P_L, λ, h, c)


# SPAD params
D = 300e-6       # Aperture
η = 0.5          # Quantum efficiency (affects both ambient and signal photons equally)
A_spad = π*(D/2)^2      # Area of SPAD aperture

# Diffuser
θ = 40          # Diffuser square pyramid apex dihedral angle (degrees)

# Scene params
r = 2.          # Target distance (m)
ρ = 0.5         # Target albedo   (dimensionless)
A_t = A(θ, r)

# Ambient power
I_A = 0.002      # W/m²
P_A = I_A*A_t

A_illum = A_t           # Illumination area is the same as the target
P_R = R(P_L, ρ, A_t, A_illum, A_spad, r, η)
N_R = N(P_R, λ, h, c)

P_AR = R(P_A, ρ, A_t, A_illum, A_spad, r, η)
N_AR = N(P_AR, λ, h, c)
SBR = N_R/N_AR

# Minimum laser power under SBR constraint
SBR_min = 5.
# SBR_min = 0.15  # Outdoor scene
P_min = I_A * 4*r^2*tand(θ/2)^2 * SBR_min
