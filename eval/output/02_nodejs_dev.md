Node.js *(JavaScript)* is a garbage collected language, so having memory leaks is possible through retainers. As Node.js applications are usually multi-tenant, business critical, and long-running, providing an accessible and efficient way of finding a memory leak is essential.

[Symptoms](#symptoms)

The user observes continuously increasing memory usage *(can be fast or slow, over days or even weeks)* then sees the process crashing and restarting by the process manager. The process is maybe running slower than before and the restarts cause some requests to fail *(load balancer responds with 502)*.

[Side Effects](#side-effects)

Process restarts due to the memory exhaustion and requests are dropped on the floor

Increased GC activity leads to higher CPU usage and slower response time

 - GC blocking the Event Loop causing slowness

Increased memory swapping slows down the process (GC activity)

May not have enough available memory to get a Heap Snapshot