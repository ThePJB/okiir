# OKIIR
## What is this?
Somewhat gives you matlab capabilities for designing z-domain filters and plotting poles and zeros, as well as IIR implementation.
I heard that IIR filters can be unstable in practice.

## Examples

## Original Inspiration
Generalized composable waveguide modeling for instruments. Eg a piano is this thing * this thing * this thing * this thing. Piano!
*IIR filters can be unstable in practice


I think they are especially bad with higher orders - eg delay of 220 for some kind of sonic tone. Nah.


Original inspiration: waveguide stuff
https://www.youtube.com/watch?v=Pugg438Cxds
the impulse sounds so goddamn good

Waveguide synthesis resource:
https://www.osar.fr/notes/waveguides/


## Where to from here?
### Go deeper
* bilinear transform, design in s domain for more stability (?)
* state-space matrix modeling
* there's probably techniques for realizing more stable IIR in practice, IDK
    * Biquads ???
    * Direct form type 2 (this is type 1) (type 2 is waaay better)
    * cascading structures
    https://www.brainkart.com/article/Structures-For-IIR-Systems_13041/
    * DIRECT REALIZATION OF POLES/ZEROS INTO CASCADE FORM
        * theres a choice of how to order them
        * what happens if I use the same transfer function structure only its a different access pattern of the coefficients? and u just put down in pole-zero form. They seem like related seuqences
        * or yeh could use biquad for the backend of it

### Go not as deep
FIR is an abstraction that won't hurt you, probably compose in code several FIR systems and explicitly do feedback yourself


### In Summary
* Interactive pole zero editor would be great for building intuition
* 3 contenders for increased stability:
    * Direct form 2 promising stability-wise
    * My idea for cascaded single units. one pole one zero units.
    * What about entirely parallel units? thats weird right. But theres parallel biquads and series biquads.
    * Biquad library
    * maybe you can transform coefficients as well as pole locations with bilinear transform?


### Topology Ideas
* Parallel poles
* Series poles
* Series with stability enforcing guardrails / nonlinearity (kinda like a mean-reverting process)

### Random Cooked Thoughts
* Poles are connected to eigenvalues
* All pole form == diagonal matrix? Probably yet diagonal form of the state matrix
* Poles connection to mandelbrot set??? Not as much because the term gets squared, not a linear combination of itself.

### Pole Analysis
* Poles at Nth roots of unity
* Conjugate poles or not

### Upcoming Features
* Interactive pole/zero editor
* novel pole cascader / serieser / paralleler