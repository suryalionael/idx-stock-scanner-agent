// Cloudflare Worker — fires at 06:00 WIB and triggers the GitHub
// `send-alert.yml` workflow via a repository_dispatch event.
//
// Secret (set once):  wrangler secret put GH_PAT
//   → fine-grained PAT, THIS repo only, Repository permission "Contents: Read and write".
// Vars are in wrangler.toml. Dispatch returns HTTP 204 on success.

export default {
  async scheduled(event, env, ctx) {
    const url = `https://api.github.com/repos/${env.GH_OWNER}/${env.GH_REPO}/dispatches`;
    const res = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.GH_PAT}`,
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
        "User-Agent": "idx-morning-trigger",
      },
      body: JSON.stringify({ event_type: env.EVENT_TYPE || "morning-alert" }),
    });
    if (res.status !== 204) {
      // Surfaces in `wrangler tail`; non-204 means the PAT/scope/repo is wrong.
      console.error(`dispatch failed: ${res.status} ${await res.text()}`);
    } else {
      console.log("dispatch ok (204) — send-alert.yml triggered");
    }
  },

  // Optional: visit the Worker URL in a browser to fire a manual test dispatch.
  async fetch(request, env, ctx) {
    await this.scheduled(null, env, ctx);
    return new Response("dispatch sent\n");
  },
};
