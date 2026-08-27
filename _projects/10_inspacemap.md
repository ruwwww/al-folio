---
layout: page
title: InSpaceMap Platform
description: Full-stack indoor venue mapping, drag-and-drop navigation graph editor, and wayfinding platform
img: assets/img/11.jpg
importance: 1
category: Full-Stack & Interactive Applications
github: https://github.com/ruwwww/inspacemap-be
---

**InSpaceMap** is a full-stack indoor venue mapping, path routing, and navigation system spanning a high-performance backend, an interactive web graph editor, and a cross-platform mobile wayfinding application.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        <a href="https://github.com/ruwwww/inspacemap-be" target="_blank" class="btn btn-primary btn-sm">Backend (Go)</a>
        <a href="https://github.com/ruwwww/inspacemap-tenant-fe" target="_blank" class="btn btn-outline-primary btn-sm ml-2">Tenant Portal (Next.js 16)</a>
        <a href="https://github.com/ruwwww/inspacemap-mobile-fe" target="_blank" class="btn btn-outline-secondary btn-sm ml-2">Mobile Client (Flutter)</a>
    </div>
</div>

### Platform Architecture

- **Backend API (`inspacemap-be`):** High-concurrency RESTful API built in **Go (Fiber)** with PostgreSQL spatial querying, MinIO S3 floorplan storage, JWT authentication, and graph pathfinding algorithms.
- **Tenant Web Portal (`inspacemap-tenant-fe`):** Interactive web application built with **Next.js 16** and TypeScript featuring a **drag-and-drop canvas graph editor** for venue administrators to construct walkable navigation graphs, place points of interest (POIs), and manage multi-venue hierarchies.
- **Mobile Navigation (`inspacemap-mobile-fe`):** Cross-platform **Flutter** mobile client offering real-time indoor turn-by-turn routing and interactive map exploration.
