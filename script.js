const scenes = [
  ["S1A", "2020-01-12", "05:32 UTC", "S1A_IW_GRDH_1SDV_20200112T053228_20200112T053253_030763_03871C_B8E9.dim_wind_gray.png"],
  ["S1A", "2020-04-15", "05:49 UTC", "S1A_IW_GRDH_1SDV_20200415T054926_20200415T054951_032134_03B6F4_48A2.dim_wind_gray.png"],
  ["S1A", "2020-06-28", "10:04 UTC", "S1A_IW_GRDH_1SDV_20200628T100410_20200628T100435_033216_03D91F_B285.dim_wind_gray.png"],
  ["S1A", "2020-11-14", "09:55 UTC", "S1A_IW_GRDH_1SDV_20201114T095523_20201114T095548_035243_041D99_BAA6.dim_wind_gray.png"],
  ["S1B", "2020-06-13", "05:57 UTC", "S1B_IW_GRDH_1SDV_20200613T055707_20200613T055732_022011_029C64_C232.dim_wind_gray.png"],
  ["S1A", "2021-03-25", "17:33 UTC", "S1A_IW_GRDH_1SDV_20210325T173319_20210325T173344_037158_04600A_9FC8.dim_wind_gray.png"],
  ["S1A", "2021-07-23", "17:33 UTC", "S1A_IW_GRDH_1SDV_20210723T173325_20210723T173350_038908_04974A_7CA3.dim_wind_gray.png"],
  ["S1A", "2021-11-10", "06:06 UTC", "S1A_IW_GRDH_1SDV_20211110T060624_20211110T060649_040505_04CDA8_A89E.dim_wind_gray.png"],
  ["S1B", "2021-09-08", "17:41 UTC", "S1B_IW_GRDH_1SDV_20210908T174128_20210908T174153_028610_036A08_3771.dim_wind_gray.png"],
  ["S1A", "2022-02-07", "17:25 UTC", "S1A_IW_GRDH_1SDV_20220207T172543_20220207T172608_041810_04F9FD_6832.dim_wind_gray.png"],
  ["S1A", "2022-05-21", "17:17 UTC", "S1A_IW_GRDH_1SDV_20220521T171740_20220521T171805_043312_052C17_EDF2.dim_wind_gray.png"],
  ["S1A", "2022-09-06", "17:17 UTC", "S1A_IW_GRDH_1SDV_20220906T171747_20220906T171812_044887_055C8B_6908.dim_wind_gray.png"],
];

const sceneGrid = document.querySelector("#scene-grid");
const isChinese = document.documentElement.lang.toLowerCase().startsWith("zh");
const lightbox = document.querySelector("#lightbox");
const lightboxImage = lightbox.querySelector("img");
const lightboxCaption = lightbox.querySelector("p");

function openLightbox(src, title) {
  lightboxImage.src = src;
  lightboxImage.alt = title;
  lightboxCaption.textContent = title;
  lightbox.showModal();
  document.body.classList.add("is-locked");
}

function buildScenes() {
  const fragment = document.createDocumentFragment();
  scenes.forEach(([satellite, date, time, filename], index) => {
    const article = document.createElement("article");
    article.className = "scene-card";
    article.dataset.satellite = satellite;
    const title = `${satellite} · ${date} ${time}`;
    const sceneDescription = isChinese ? "SAR 近地面风速场" : "SAR near-surface wind field";
    const enlargeLabel = isChinese ? `放大 ${title}` : `Enlarge ${title}`;
    article.innerHTML = `
      <button class="image-button" aria-label="${enlargeLabel}">
        <img src="assets/gallery/${filename}" alt="${title} ${sceneDescription}" loading="${index < 3 ? "eager" : "lazy"}" />
      </button>
      <div class="scene-meta">
        <div><b>${date}</b><small>${time}</small></div>
        <span class="scene-tag">${satellite}</span>
      </div>`;
    article.querySelector("button").addEventListener("click", () => {
      openLightbox(`assets/gallery/${filename}`, title);
    });
    fragment.append(article);
  });
  sceneGrid.append(fragment);
}

buildScenes();

document.querySelectorAll("[data-lightbox]").forEach((button) => {
  button.addEventListener("click", () => {
    openLightbox(button.dataset.lightbox, button.dataset.title);
  });
});

document.querySelectorAll(".scene-filter").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelector(".scene-filter.is-active")?.classList.remove("is-active");
    button.classList.add("is-active");
    document.querySelectorAll(".scene-card").forEach((card) => {
      card.classList.toggle(
        "is-hidden",
        button.dataset.filter !== "all" && card.dataset.satellite !== button.dataset.filter
      );
    });
  });
});

function closeLightbox() {
  lightbox.close();
  document.body.classList.remove("is-locked");
  lightboxImage.src = "";
}

lightbox.querySelector(".lightbox-close").addEventListener("click", closeLightbox);
lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) closeLightbox();
});
lightbox.addEventListener("close", () => document.body.classList.remove("is-locked"));

const navToggle = document.querySelector(".nav-toggle");
const nav = document.querySelector("#main-nav");
navToggle.addEventListener("click", () => {
  const open = nav.classList.toggle("is-open");
  navToggle.setAttribute("aria-expanded", String(open));
});
nav.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    nav.classList.remove("is-open");
    navToggle.setAttribute("aria-expanded", "false");
  });
});
