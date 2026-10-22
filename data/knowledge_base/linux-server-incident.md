# Linux and Server Incidents.
If a Linux server becomes unresponsive, inspect load average, CPU utilization, memory pressure, disk usage, and process states before restarting it.

If the filesystem reaches 100% usage, identify large files, log growth, temporary files, and deleted files still held open by processes before removing data.

If disk space appears available but applications cannot create files, check inode availability, filesystem permissions, quotas, and read-only mount status.

If a service fails to start, inspect its systemd status and journal logs to identify configuration errors, missing dependencies, or permission problems.

If a process consumes excessive CPU, identify the process and examine its workload before terminating it.

If SSH access fails, verify network connectivity, the SSH service, firewall rules, authentication configuration, and recent security changes.

Never disable the firewall permanently as a first response to a connectivity problem.

If a server's clock differs significantly from other systems, verify time synchronization because incorrect timestamps can complicate log correlation, authentication, and distributed tracing.