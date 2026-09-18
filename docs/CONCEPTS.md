# LayerCake concepts

LayerCake separates model execution, capability packaging, acquisition, and
scientific evidence. Keeping these terms distinct prevents most integration
mistakes.

## Core vocabulary

| Term | Meaning | Not the same as |
| --- | --- | --- |
| **Execution host** | Runtime code that validates artifacts, maintains state, invokes the core or selected cakes, and returns UTF-8 output. | A trained English model or an acquisition system. |
| **English core** | The model artifact responsible for domain-independent language behavior in a deployment. | The Python, SQL, or other specialist cakes. |
| **Cake** | An immutable capability package containing declarative metadata and closed-schema tensor payloads. | Executable plug-in code, a prompt template, or a claim of quality. |
| **Cake ABI** | The versioned structural and behavioral contract between a host and a cake. | ABI's foreign-teacher extraction research. |
| **Registry** | Local content-addressed storage plus the active installed-version records. | A remote download service. |
| **Catalog** | Declarative records describing packages that may be discovered. | Proof that a package is installed, authenticated, or useful. |
| **Router** | A policy or model that selects eligible installed cakes for a request. | A mechanism allowed to bypass package authentication. |
| **Orchestrator** | Coordinates core-only, selected-cake, top-k, or structured multidomain execution. | Latent neural fusion unless that mode is separately implemented and certified. |
| **Acquisition** | Producing a capability artifact through training, extraction, or another process. | Hosting or installing the resulting artifact. |
| **Evidence lineage** | Exact code, data, artifact, runtime, hardware, observations, and verifier identities supporting a claim. | A filename, branch label, or manually written status. |

## Package identity versus behavior

LayerCake treats these as separate questions:

1. **Package identity:** are the archive, manifest, and tensor bytes exactly the
   authenticated bytes expected by the host?
2. **Compatibility:** does the package declare the ABI, precision, backend, and
   required features this host supports?
3. **Semantic behavior:** does the installed package retain its intended
   capability on a frozen functional suite?
4. **Performance:** does that same artifact meet latency, memory, and throughput
   gates on the declared hardware?

Passing one question does not imply the others.

## LayerCake and ABI

The repositories share a handoff boundary, not an implementation tree:

```text
foreign source model
  -> ABI extraction, labeling, minimization, and artifact validation
  -> immutable external capability artifact
  -> LayerCake compatibility and conformance checks
  -> signed cake/core package
  -> LayerCake installation, routing, and execution
```

LayerCake contains no foreign-teacher extraction implementation or evidence.
An artifact produced elsewhere earns no LayerCake quality or performance claim
until the exact final artifact is tested on the exact host lineage.

## Sealed release versus current development

The tag `layercake-moonshot-final` is the frozen scientific release. The
default branch contains later host constructs and deliberately does not inherit
the tag's affected certificates. This distinction is explained in
[Project status](PROJECT_STATUS.md).
