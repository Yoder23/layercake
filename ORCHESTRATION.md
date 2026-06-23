# Orchestration Interface

`layercake.orchestration.HandoffPacket` is the local CorticalSwarm integration contract.
It serializes claims, evidence, uncertainty, ABI version, input/patching mode, active
bricks, optional compressed ABI state, and parent hash. The packet hash covers canonical
JSON and rejects mutation.

`LayerCakeOrchestrator` is a deterministic simulation stub. It selects the cheapest
available model below the uncertainty threshold and a larger verifier above it, while
activating only ABI-compatible domain bricks. Network transport, authentication, policy,
and learned routing belong in CorticalSwarm and are not implemented here.
